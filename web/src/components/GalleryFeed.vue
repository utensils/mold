<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { GalleryImage } from "../types";
import GalleryCard from "./GalleryCard.vue";

type ViewMode = "feed" | "grid";

const props = defineProps<{
  entries: GalleryImage[];
  loading: boolean;
  view: ViewMode;
  muted: boolean;
}>();

const emit = defineEmits<{
  (e: "open", item: GalleryImage): void;
}>();

/*
 * Chunked rendering
 * -----------------
 * The gallery commonly holds 1000+ items. Mounting all of them up front
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
        if (entry.isIntersecting && hasMore.value) {
          loadMore();
        }
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

// Reset the visible window whenever the filtered list or view mode
// changes — narrowing the set shouldn't keep 1500 cards mounted, and a
// switch from grid to feed (or vice versa) should snap back to page 1.
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
</script>

<template>
  <section>
    <!-- Loading skeletons -->
    <div
      v-if="skeletons > 0 && view === 'grid'"
      class="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5"
    >
      <div
        v-for="i in skeletons"
        :key="`skel-${i}`"
        class="aspect-square animate-pulse rounded-2xl bg-white/[0.04]"
      ></div>
    </div>
    <div v-else-if="skeletons > 0" class="flex flex-col gap-6">
      <div
        v-for="i in skeletons"
        :key="`skel-${i}`"
        class="aspect-[4/3] w-full animate-pulse rounded-3xl bg-white/[0.04]"
      ></div>
    </div>

    <!-- Empty state -->
    <div
      v-else-if="entries.length === 0"
      class="glass flex flex-col items-center gap-3 rounded-[var(--radius-card)] px-6 py-20 text-center"
    >
      <svg
        class="h-14 w-14 text-ink-400"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.6"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <rect x="3" y="5" width="18" height="14" rx="3" />
        <circle cx="9" cy="11" r="1.6" />
        <path d="m4 19 6-6 4 4 3-3 3 3" />
      </svg>
      <div>
        <p class="text-lg font-medium text-ink-100">Nothing to show.</p>
        <p class="mt-1 text-sm text-ink-400">
          Try clearing the search or filter, or generate something with
          <code
            class="rounded bg-white/10 px-1.5 py-0.5 text-[12px] text-ink-100"
            >mold run</code
          >.
        </p>
      </div>
    </div>

    <!--
      Feed mode: Tumblr-style single-column stream.
      On mobile we cancel the parent page padding (`-mx-4 sm:mx-auto`) so
      cards run edge-to-edge; on sm+ we constrain to a comfortable reading
      width. Spacing between cards is tight on mobile (2) and generous on
      larger screens (8).
    -->
    <div
      v-else-if="view === 'feed'"
      class="-mx-4 flex flex-col gap-2 sm:-mx-0 sm:mx-auto sm:max-w-3xl sm:gap-8 lg:max-w-4xl"
    >
      <GalleryCard
        v-for="entry in visibleEntries"
        :key="entry.filename"
        :item="entry"
        :muted="muted"
        variant="feed"
        @open="emit('open', entry)"
      />
    </div>

    <!-- Grid mode: dense masonry via CSS columns for scanning. -->
    <div
      v-else
      class="columns-2 gap-4 sm:columns-3 md:columns-4 lg:columns-5 xl:columns-6 [&>*]:mb-4 [&>*]:break-inside-avoid"
    >
      <GalleryCard
        v-for="entry in visibleEntries"
        :key="entry.filename"
        :item="entry"
        :muted="muted"
        variant="grid"
        @open="emit('open', entry)"
      />
    </div>

    <!-- Load-more sentinel -->
    <div
      v-if="entries.length > 0"
      ref="sentinel"
      class="mt-8 flex h-10 items-center justify-center text-[12px] font-medium text-ink-400"
      aria-hidden="true"
    >
      <span v-if="hasMore">
        Loading more… ({{ visibleCount }}/{{ entries.length }})
      </span>
      <span v-else class="opacity-60">
        End of feed · {{ entries.length }} items
      </span>
    </div>
  </section>
</template>
