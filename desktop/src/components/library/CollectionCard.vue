<script setup lang="ts">
/*
 * CollectionCard — one 150px card in the My images album strip: a single
 * 74px cover, the name, and a mono count. The machines holding it and when it
 * last changed are the tooltip's, not the card's — the strip sits above the
 * grid and must stay one row tall. Pure: the parent resolves media
 * paths/targets and owns every action (open = click / Enter, menu =
 * right-click).
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
    /** Cover thumbnails, set cover first; only the first is drawn. */
    covers: readonly CoverTile[];
    hidden?: boolean;
    /** Optional clock for deterministic tests. */
    nowMs?: number | undefined;
  }>(),
  { updatedAt: null },
);

const emit = defineEmits<{ open: []; contextmenu: [event: MouseEvent] }>();

const meta = computed(() => `${props.count} ${props.count === 1 ? "picture" : "pictures"}`);

/** Everything the card no longer shows, kept where a pointer can still ask. */
const tooltip = computed(() => {
  const parts = [props.name];
  if (props.hostLabels.length > 0) parts.push(props.hostLabels.join(" · "));
  if (props.updatedAt != null) {
    parts.push(`Updated ${timeAgo(props.updatedAt * 1000, props.nowMs ?? Date.now())}`);
  }
  return parts.join(" · ");
});

const cover = computed(() => props.covers[0] ?? null);
</script>

<template>
  <button
    type="button"
    class="ms-ccard group flex flex-col gap-2 rounded-control border border-border bg-panel p-2.5 text-left transition-colors duration-100 hover:border-border-focus hover:bg-surface focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
    data-test="collection-card"
    :aria-label="`Open album ${name}`"
    :title="tooltip"
    @click="emit('open')"
    @contextmenu="emit('contextmenu', $event)"
  >
    <span
      class="ms-ccard__cover flex h-[74px] w-full items-center justify-center overflow-hidden border border-border bg-media-bed"
      data-test="collection-cover"
    >
      <AuthedMedia
        v-if="cover"
        :path="cover.path"
        :target="cover.target"
        :cache-key="cover.cacheKey"
        :media-version="cover.mediaVersion ?? null"
        :video="cover.video === true"
        :alt="cover.alt ?? ''"
        class="h-full w-full object-cover"
      />
      <Icon v-else name="collection" :size="22" :stroke-width="1.4" class="text-fg-dim" />
    </span>
    <span class="truncate text-xs font-semibold text-fg" data-test="collection-name">
      {{ name }}
      <span
        v-if="hidden"
        class="ml-1.5 font-mono text-micro font-medium uppercase text-fg-dim"
        data-test="collection-hidden-badge"
        >Hidden</span
      >
    </span>
    <span class="font-mono text-micro text-fg-dim" data-test="collection-meta">{{ meta }}</span>
  </button>
</template>

<style scoped>
.ms-ccard__cover :deep(img),
.ms-ccard__cover :deep(video) {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
</style>
