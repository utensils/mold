<script setup lang="ts">
/*
 * CollectionCard — one shelf card in the Library's Collections scope (V3
 * "Shelf"): a 2×2 cover mosaic (the set cover first, then the newest prints),
 * display-weight name, a mono meta line (count · hosts), and last-updated.
 * Pure: the parent resolves media paths/targets and owns every action
 * (open = click / Enter, menu = right-click).
 */
import { computed } from "vue";
import Icon from "@ui/components/Icon.vue";
import AuthedMedia from "../gallery/AuthedMedia.vue";
import type { ApiTarget } from "../../lib/api/client";
import { timeAgo } from "../../lib/format";

export interface CoverTile {
  path: string;
  target: ApiTarget | null;
  cacheKey: string | null;
  /** Content version so the cover can come from the persistent native cache. */
  mediaVersion?: string | null;
  video?: boolean;
  alt?: string;
}

const props = withDefaults(
  defineProps<{
    name: string;
    /** Logical prints in the collection. */
    count: number;
    /** Host labels holding the collection ("This Mac", "plato"). */
    hostLabels: readonly string[];
    /** Unix seconds of the latest change across hosts; null = unknown. */
    updatedAt?: number | null;
    /** Up to four cover thumbnails, set cover first. */
    covers: readonly CoverTile[];
    hidden?: boolean;
    /** Optional clock for deterministic tests. */
    nowMs?: number | undefined;
  }>(),
  { updatedAt: null },
);

const emit = defineEmits<{ open: []; contextmenu: [event: MouseEvent] }>();

const meta = computed(() => {
  const noun = props.count === 1 ? "print" : "prints";
  const hosts = props.hostLabels.join(" · ");
  return hosts ? `${props.count} ${noun} · ${hosts}` : `${props.count} ${noun}`;
});

const updated = computed(() =>
  props.updatedAt != null
    ? `Updated ${timeAgo(props.updatedAt * 1000, props.nowMs ?? Date.now())}`
    : "",
);

const mosaic = computed(() => props.covers.slice(0, 4));
</script>

<template>
  <button
    type="button"
    class="ms-ccard group flex flex-col gap-0.5 rounded-card border border-edge bg-bench p-2.5 text-left transition-[transform,box-shadow] duration-100 hover:-translate-y-0.5 hover:shadow-[0_10px_24px_rgba(0,0,0,0.25)] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-safelight"
    data-test="collection-card"
    :aria-label="`Open collection ${name}`"
    @click="emit('open')"
    @contextmenu="emit('contextmenu', $event)"
  >
    <span
      class="ms-ccard__mosaic mb-2 grid aspect-[4/3] w-full overflow-hidden rounded-[8px] bg-print-surface"
      :class="mosaic.length <= 1 ? 'grid-cols-1' : 'grid-cols-2 gap-0.5'"
      data-test="collection-mosaic"
    >
      <span v-if="mosaic.length === 0" class="flex items-center justify-center text-ink-3">
        <Icon name="collection" :size="26" :stroke-width="1.4" />
      </span>
      <AuthedMedia
        v-for="(cover, i) in mosaic"
        :key="`${cover.cacheKey ?? ''}:${cover.path}:${i}`"
        :path="cover.path"
        :target="cover.target"
        :cache-key="cover.cacheKey"
        :media-version="cover.mediaVersion ?? null"
        :video="cover.video === true"
        :alt="cover.alt ?? ''"
        class="h-full w-full object-cover"
      />
    </span>
    <span class="font-display text-[15px] font-semibold text-ink" data-test="collection-name">
      {{ name }}
      <span
        v-if="hidden"
        class="ml-1.5 font-utility text-[9.5px] font-medium uppercase text-ink-3"
        data-test="collection-hidden-badge"
        >Hidden</span
      >
    </span>
    <span class="font-utility text-[10.5px] text-ink-3" data-test="collection-meta">{{
      meta
    }}</span>
    <span v-if="updated" class="text-[11.5px] text-ink-3" data-test="collection-updated">
      {{ updated }}
    </span>
  </button>
</template>

<style scoped>
.ms-ccard__mosaic :deep(img),
.ms-ccard__mosaic :deep(video) {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
</style>
