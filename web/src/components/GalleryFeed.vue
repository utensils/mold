<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { GalleryImage } from "../types";
import { useTweaks } from "../composables/useTweaks";
import GalleryCard from "./GalleryCard.vue";

type ViewMode = "feed" | "grid";

const props = defineProps<{
  entries: GalleryImage[];
  loading: boolean;
  view: ViewMode;
  muted: boolean;
  starred?: Set<string>;
}>();

const emit = defineEmits<{
  (e: "open", item: GalleryImage): void;
  (e: "star", item: GalleryImage): void;
  (e: "rerun", item: GalleryImage): void;
}>();

const { tweaks } = useTweaks();

/*
 * Chunked rendering
 * -----------------
 * The gallery commonly holds 1000+ items. Mounting them all up front
 * would stall the main thread (every card registers an IntersectionObserver
 * and a <video>/<img> element). Feed mode (tall cards) shows fewer per
 * viewport so we render a smaller chunk; grid mode packs more, so we
 * render more per chunk.
 */
const pageSize = computed(() => (props.view === "feed" ? 40 : 150));
const visibleCount = ref(pageSize.value);
const sentinel = ref<HTMLElement | null>(null);

const visibleEntries = computed(() =>
  props.entries.slice(0, visibleCount.value),
);
const hasMore = computed(() => visibleCount.value < props.entries.length);

function loadMore() {
  visibleCount.value = Math.min(
    visibleCount.value + pageSize.value,
    props.entries.length,
  );
}

let observer: IntersectionObserver | null = null;

function installObserver() {
  observer?.disconnect();
  if (!sentinel.value) return;
  observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting && hasMore.value) loadMore();
      }
    },
    { rootMargin: "800px 0px" },
  );
  observer.observe(sentinel.value);
}

onMounted(() => {
  installObserver();
});

onBeforeUnmount(() => {
  observer?.disconnect();
});

watch(
  () => [props.entries, props.view] as const,
  () => {
    visibleCount.value = Math.min(pageSize.value, props.entries.length);
    queueMicrotask(installObserver);
  },
);

const skeletons = computed(() =>
  props.loading && props.entries.length === 0 ? 8 : 0,
);

function isStarred(item: GalleryImage): boolean {
  return props.starred ? props.starred.has(item.filename) : false;
}
</script>

<template>
  <section>
    <!-- Loading skeletons -->
    <div
      v-if="skeletons > 0 && view === 'grid'"
      class="grid-wrap"
      :class="`density-${tweaks.density}`"
    >
      <div
        v-for="i in skeletons"
        :key="`skel-${i}`"
        class="card-skeleton"
        style="aspect-ratio: 1 / 1"
      />
    </div>
    <div v-else-if="skeletons > 0" class="feed-wrap">
      <div
        v-for="i in skeletons"
        :key="`skel-${i}`"
        class="card-skeleton"
        style="aspect-ratio: 4 / 3; width: 100%"
      />
    </div>

    <!-- Empty state -->
    <div v-else-if="entries.length === 0" class="empty glass-panel">
      <svg
        width="40"
        height="40"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.6"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <rect x="3" y="5" width="18" height="14" rx="2" />
        <circle cx="9" cy="11" r="1.6" />
        <path d="m4 19 6-6 4 4 3-3 3 3" />
      </svg>
      <p class="empty-title">Nothing to show.</p>
      <p class="empty-sub">
        Try clearing the search or filter, or generate something with
        <code>mold run</code>.
      </p>
    </div>

    <!-- Feed — single-column stream -->
    <div v-else-if="view === 'feed'" class="feed-wrap">
      <GalleryCard
        v-for="entry in visibleEntries"
        :key="entry.filename"
        :item="entry"
        :muted="muted"
        :starred="isStarred(entry)"
        variant="feed"
        @open="emit('open', entry)"
        @star="emit('star', entry)"
        @rerun="emit('rerun', entry)"
      />
    </div>

    <!-- Grid — dense masonry via CSS columns -->
    <div v-else class="grid-wrap" :class="`density-${tweaks.density}`">
      <GalleryCard
        v-for="entry in visibleEntries"
        :key="entry.filename"
        :item="entry"
        :muted="muted"
        :starred="isStarred(entry)"
        variant="grid"
        @open="emit('open', entry)"
        @star="emit('star', entry)"
        @rerun="emit('rerun', entry)"
      />
    </div>

    <!-- Load-more sentinel -->
    <div
      v-if="entries.length > 0"
      ref="sentinel"
      class="mt-8 flex h-10 items-center justify-center"
      style="font-size: 12px; color: var(--fg-3)"
      aria-hidden="true"
    >
      <span v-if="hasMore">
        Loading more… ({{ visibleCount }}/{{ entries.length }})
      </span>
      <span v-else style="opacity: 0.6">
        End of feed · {{ entries.length }} items
      </span>
    </div>
  </section>
</template>
