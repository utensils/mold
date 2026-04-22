<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { GalleryImage } from "../types";
import GalleryCard from "./GalleryCard.vue";

type ViewMode = "feed" | "grid";

// Selection / hide-mode props are gallery-only. The Generate page reuses
// this component for its small inline preview list and doesn't need any
// of it — defaults keep those surfaces behaviorally identical to before.
const props = withDefaults(
  defineProps<{
    entries: GalleryImage[];
    loading: boolean;
    view: ViewMode;
    muted: boolean;
    selectMode?: boolean;
    selection?: Set<string>;
    hideMode?: boolean;
    revealed?: Set<string>;
  }>(),
  {
    selectMode: false,
    selection: () => new Set<string>(),
    hideMode: false,
    revealed: () => new Set<string>(),
  },
);

const emit = defineEmits<{
  (e: "open", item: GalleryImage): void;
  (
    e: "toggle-select",
    payload: { item: GalleryImage; shift: boolean; meta: boolean },
  ): void;
  (e: "reveal", item: GalleryImage): void;
  // Drag-select: emitted with the full finalized selection the parent
  // should adopt. We snapshot the starting selection at pointerdown so
  // additive drags (shift/meta) merge cleanly with the existing set;
  // plain drags replace it. The parent just assigns whatever arrives.
  (e: "drag-select", payload: { filenames: string[] }): void;
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

/*
 * Drag / marquee selection
 * ------------------------
 * Active only when `selectMode` is true. We capture pointerdown on the
 * feed root, draw a translucent rectangle as the pointer moves, and
 * diff it against each mounted card's bounding rect to derive the set
 * of filenames touched by the marquee.
 *
 * Notes:
 *  - We gate on a 6px movement threshold so a simple click doesn't
 *    register as a drag and wipe the selection.
 *  - `shift` / `meta` during drag = additive (union with the current
 *    selection). A bare drag replaces the selection with the marquee.
 *  - Scroll-while-dragging is out of scope for v1; the feed already
 *    renders ~150 cards at a time and they all fit in the mouseable area.
 */
const feedRoot = ref<HTMLElement | null>(null);
const dragBox = ref<{
  x: number;
  y: number;
  w: number;
  h: number;
} | null>(null);

type DragState = {
  startX: number;
  startY: number;
  additive: boolean;
  started: boolean;
  base: Set<string>;
};
let drag: DragState | null = null;

function onPointerDown(evt: PointerEvent) {
  if (!props.selectMode) return;
  // Left button only; ignore right-clicks and touch-pinch-zoom gestures.
  if (evt.button !== 0) return;
  // Don't start a drag if the click originated on a card — we want the
  // click handler on the card to toggle its selection cleanly. Empty
  // space (gaps between cards, padding around the grid) is the drag
  // surface.
  const target = evt.target as HTMLElement | null;
  if (target?.closest("[data-filename]")) return;
  if (target?.closest("button, a, input, textarea, [data-swipe-ignore]")) {
    return;
  }
  drag = {
    startX: evt.clientX,
    startY: evt.clientY,
    additive: evt.shiftKey || evt.metaKey || evt.ctrlKey,
    started: false,
    // Snapshot the selection at drag start so additive drags union cleanly.
    // A plain drag discards this and emits marquee-only.
    base: new Set(props.selection),
  };
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp, { once: true });
}

function onPointerMove(evt: PointerEvent) {
  if (!drag) return;
  const dx = evt.clientX - drag.startX;
  const dy = evt.clientY - drag.startY;
  if (!drag.started && Math.hypot(dx, dy) < 6) return;
  drag.started = true;
  const x = Math.min(drag.startX, evt.clientX);
  const y = Math.min(drag.startY, evt.clientY);
  const w = Math.abs(dx);
  const h = Math.abs(dy);
  dragBox.value = { x, y, w, h };
  const hits = collectHits(x, y, w, h);
  const final = drag.additive
    ? new Set([...drag.base, ...hits])
    : new Set(hits);
  emit("drag-select", { filenames: Array.from(final) });
}

function onPointerUp() {
  window.removeEventListener("pointermove", onPointerMove);
  drag = null;
  dragBox.value = null;
}

function collectHits(x: number, y: number, w: number, h: number): string[] {
  if (!feedRoot.value) return [];
  const cards = feedRoot.value.querySelectorAll<HTMLElement>("[data-filename]");
  const hits: string[] = [];
  const right = x + w;
  const bottom = y + h;
  for (const card of cards) {
    const rect = card.getBoundingClientRect();
    // AABB intersection test in viewport coords.
    if (
      rect.right < x ||
      rect.left > right ||
      rect.bottom < y ||
      rect.top > bottom
    ) {
      continue;
    }
    const name = card.dataset.filename;
    if (name) hits.push(name);
  }
  return hits;
}

onBeforeUnmount(() => {
  window.removeEventListener("pointermove", onPointerMove);
});
</script>

<template>
  <section
    ref="feedRoot"
    :class="{ 'select-none': selectMode }"
    @pointerdown="onPointerDown"
  >
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
        :select-mode="selectMode"
        :selected="selection.has(entry.filename)"
        :hide-mode="hideMode"
        :revealed="revealed.has(entry.filename)"
        @open="emit('open', entry)"
        @toggle-select="emit('toggle-select', $event)"
        @reveal="emit('reveal', $event)"
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
        :select-mode="selectMode"
        :selected="selection.has(entry.filename)"
        :hide-mode="hideMode"
        :revealed="revealed.has(entry.filename)"
        @open="emit('open', entry)"
        @toggle-select="emit('toggle-select', $event)"
        @reveal="emit('reveal', $event)"
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

    <!-- Marquee rectangle drawn during drag-select. Positioned in the
         fixed viewport because drag coords come from pointer events. -->
    <div
      v-if="dragBox"
      class="pointer-events-none fixed z-40 rounded-sm border border-brand-400/70 bg-brand-400/10"
      :style="{
        left: `${dragBox.x}px`,
        top: `${dragBox.y}px`,
        width: `${dragBox.w}px`,
        height: `${dragBox.h}px`,
      }"
      aria-hidden="true"
    ></div>
  </section>
</template>
