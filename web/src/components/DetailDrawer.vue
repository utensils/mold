<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { imageUrl, thumbnailUrl } from "../api";
import type { GalleryImage } from "../types";
import { mediaKind } from "../types";
import Metadata from "./Metadata.vue";

const props = defineProps<{
  item: GalleryImage | null;
  hasPrev: boolean;
  hasNext: boolean;
  // Position in the currently-filtered list — surfaced as "N / total" in
  // the top bar so the viewer has a sense of place in a 1000-item feed.
  index: number;
  total: number;
  // When false (the default on a vanilla server), the drawer renders in
  // read-only mode — the destructive delete affordance is hidden entirely
  // rather than inviting a 403 from the backend.
  canDelete: boolean;
  // Global audio preference — the drawer honours it for consistency with
  // the feed, but also shows the native `controls` so the user can
  // override per item.
  muted: boolean;
}>();

const emit = defineEmits<{
  (e: "close"): void;
  (e: "prev"): void;
  (e: "next"): void;
  (e: "delete", item: GalleryImage): void;
}>();

const copied = ref<false | "prompt" | "seed">(false);
let copyTimer: ReturnType<typeof setTimeout> | null = null;

// Metadata sheet visibility (mobile-first). Default closed so the media
// fills the viewport; user pops the sheet open via the info button or a
// swipe up from the bottom edge. On desktop the sheet is always visible
// as a fixed right-hand pane and this flag is ignored.
const sheetOpen = ref(false);

const kind = computed(() =>
  props.item ? mediaKind(props.item.format, props.item.filename) : "image",
);

const mediaSrc = computed(() =>
  props.item ? imageUrl(props.item.filename) : "",
);
const thumbSrc = computed(() =>
  props.item ? thumbnailUrl(props.item.filename) : "",
);

// ── Swipe navigation ─────────────────────────────────────────────────────
//
// Mobile viewers navigate the filtered list with vertical swipes — swipe
// **down** to advance to the next (older) item, swipe **up** to go back
// to the previous (newer) one. This mirrors the direction of travel in
// the feed itself: scrolling the feed downwards reveals older items, and
// a swipe-down in the detail view continues that motion.
//
// We attach the touch listeners to the media pane only, so the metadata
// sheet is still swipe-scrollable without hijacking navigation.

const SWIPE_THRESHOLD_PX = 60;
const SWIPE_MAX_HORIZONTAL_DRIFT = 80;

const touchStartY = ref<number | null>(null);
const touchStartX = ref<number | null>(null);
const touchDeltaY = ref(0);

/*
 * Decide whether a touch that originated on `target` should be treated as a
 * navigation swipe. We skip anything inside the metadata sheet (the user is
 * trying to scroll its contents), the top-bar buttons (they need their
 * normal tap behaviour), and native `<video>` elements (tapping the video
 * should engage the default controls — the swipe threshold is high enough
 * that short taps won't be mistaken for navigation anyway, but we'd rather
 * lose the one-in-a-thousand long swipe on a video than break every tap).
 */
function shouldIgnoreTouch(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  return !!target.closest("[data-swipe-ignore]");
}

function onTouchStart(e: TouchEvent) {
  const t = e.touches[0];
  if (!t || shouldIgnoreTouch(e.target)) {
    touchStartY.value = null;
    return;
  }
  touchStartY.value = t.clientY;
  touchStartX.value = t.clientX;
  touchDeltaY.value = 0;
}

function onTouchMove(e: TouchEvent) {
  const t = e.touches[0];
  if (!t || touchStartY.value === null) return;
  touchDeltaY.value = t.clientY - touchStartY.value;
}

function onTouchEnd(e: TouchEvent) {
  if (touchStartY.value === null) return;
  const endTouch = e.changedTouches[0];
  const dy = touchDeltaY.value;
  const dx =
    endTouch && touchStartX.value !== null
      ? endTouch.clientX - touchStartX.value
      : 0;

  touchStartY.value = null;
  touchStartX.value = null;
  touchDeltaY.value = 0;

  // Ignore gestures that were mostly horizontal — the user might be trying
  // to scrub the video timeline, not navigate.
  if (Math.abs(dx) > SWIPE_MAX_HORIZONTAL_DRIFT) return;

  if (dy > SWIPE_THRESHOLD_PX && props.hasNext) {
    emit("next");
  } else if (dy < -SWIPE_THRESHOLD_PX && props.hasPrev) {
    emit("prev");
  }
}

function onKey(e: KeyboardEvent) {
  if (!props.item) return;
  if (e.key === "Escape") {
    if (sheetOpen.value) {
      sheetOpen.value = false;
    } else {
      emit("close");
    }
  } else if (e.key === "ArrowLeft" && props.hasPrev) emit("prev");
  else if (e.key === "ArrowRight" && props.hasNext) emit("next");
  else if ((e.key === "ArrowDown" || e.key === "j") && props.hasNext)
    emit("next");
  else if ((e.key === "ArrowUp" || e.key === "k") && props.hasPrev)
    emit("prev");
  else if (e.key === "i") sheetOpen.value = !sheetOpen.value;
}

onMounted(() => {
  window.addEventListener("keydown", onKey);
});

onBeforeUnmount(() => {
  window.removeEventListener("keydown", onKey);
  if (copyTimer) clearTimeout(copyTimer);
});

// Lock body scroll while the drawer is open so the background doesn't
// shift when we navigate between items on mobile. Also reset the
// metadata sheet every time a new item is selected — switching images
// shouldn't leave the sheet open from the previous one.
watch(
  () => props.item,
  (open, prev) => {
    if (typeof document !== "undefined") {
      document.body.style.overflow = open ? "hidden" : "";
    }
    if (!prev || !open || prev.filename !== open.filename) {
      sheetOpen.value = false;
    }
  },
);

async function copy(text: string, kind: "prompt" | "seed") {
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    copied.value = kind;
    if (copyTimer) clearTimeout(copyTimer);
    copyTimer = setTimeout(() => (copied.value = false), 1600);
  } catch {
    // swallow — some contexts block clipboard
  }
}

function confirmDelete() {
  if (!props.item) return;
  if (!window.confirm(`Delete ${props.item.filename}? This can't be undone.`))
    return;
  emit("delete", props.item);
}
</script>

<template>
  <Transition name="drawer">
    <div
      v-if="item"
      class="drawer-root fixed inset-0 z-40 flex items-stretch justify-center bg-ink-950/95 backdrop-blur-lg"
      @touchstart.passive="onTouchStart"
      @touchmove.passive="onTouchMove"
      @touchend.passive="onTouchEnd"
      @touchcancel.passive="onTouchEnd"
    >
      <!-- Top bar: close + counter + info toggle. Always visible; floats
           over the media on mobile, sits above the content on desktop.
           Marked swipe-ignore so tapping buttons here doesn't accidentally
           start a navigation gesture. -->
      <div
        data-swipe-ignore
        class="pointer-events-none absolute inset-x-0 top-0 z-20 flex items-center justify-between gap-3 bg-gradient-to-b from-black/60 via-black/30 to-transparent px-4 pb-10 pt-[max(0.75rem,env(safe-area-inset-top))] lg:from-transparent lg:via-transparent lg:pb-4"
      >
        <button
          class="pointer-events-auto inline-flex h-10 w-10 items-center justify-center rounded-full bg-white/10 text-white backdrop-blur transition hover:bg-white/20"
          aria-label="Close"
          @click="emit('close')"
        >
          <svg
            class="h-4 w-4"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2.2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path d="M6 6l12 12" />
            <path d="M18 6 6 18" />
          </svg>
        </button>

        <div
          v-if="total > 0"
          class="pointer-events-auto rounded-full bg-white/10 px-3 py-1.5 text-[11.5px] font-medium tabular-nums text-white/85 backdrop-blur"
        >
          {{ index + 1 }} / {{ total }}
        </div>

        <button
          class="pointer-events-auto inline-flex h-10 w-10 items-center justify-center rounded-full bg-white/10 text-white backdrop-blur transition hover:bg-white/20 lg:hidden"
          :aria-pressed="sheetOpen"
          aria-label="Toggle details"
          @click="sheetOpen = !sheetOpen"
        >
          <svg
            class="h-4 w-4"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2.2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <circle cx="12" cy="12" r="9" />
            <path d="M12 8v.01" />
            <path d="M11 12h1v5h1" />
          </svg>
        </button>
      </div>

      <!-- Prev / Next (desktop floating, hidden on mobile — mobile uses
           vertical swipes instead). -->
      <button
        v-if="hasPrev"
        data-swipe-ignore
        class="absolute left-4 top-1/2 z-10 hidden h-12 w-12 -translate-y-1/2 items-center justify-center rounded-full bg-white/10 text-white backdrop-blur transition hover:bg-white/20 lg:inline-flex"
        aria-label="Previous"
        @click="emit('prev')"
      >
        <svg
          class="h-5 w-5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2.2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="m15 6-6 6 6 6" />
        </svg>
      </button>
      <button
        v-if="hasNext"
        data-swipe-ignore
        class="absolute right-4 top-1/2 z-10 hidden h-12 w-12 -translate-y-1/2 items-center justify-center rounded-full bg-white/10 text-white backdrop-blur transition hover:bg-white/20 lg:inline-flex"
        aria-label="Next"
        @click="emit('next')"
      >
        <svg
          class="h-5 w-5"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2.2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path d="m9 6 6 6-6 6" />
        </svg>
      </button>

      <!-- Outer row. Media pane + side-aside on desktop, stacked on mobile.-->
      <div
        class="relative mx-auto flex h-full w-full max-w-7xl flex-col lg:flex-row"
      >
        <!-- Media pane: full viewport minus the top bar on mobile; flex-1
             of a row on desktop. Touch handlers live on the drawer root
             so that swipes anywhere — on the dark backdrop, on the image,
             or over video controls — drive navigation. We still render
             the drag-feedback transform here, driven by `touchDeltaY`. -->
        <div
          class="relative flex min-h-0 flex-1 items-center justify-center p-0 sm:p-4 lg:h-full lg:p-6"
          :style="{
            transform:
              touchDeltaY !== 0
                ? `translateY(${Math.max(Math.min(touchDeltaY * 0.3, 60), -60)}px)`
                : undefined,
            transition: touchDeltaY !== 0 ? 'none' : 'transform 0.2s ease',
          }"
          @click.self="emit('close')"
        >
          <img
            v-if="kind !== 'video'"
            :src="mediaSrc"
            :alt="item.metadata.prompt || item.filename"
            class="max-h-[100svh] max-w-full object-contain lg:max-h-[90svh] lg:rounded-2xl lg:shadow-[var(--shadow-float)]"
          />
          <video
            v-else
            data-swipe-ignore
            :src="mediaSrc"
            :poster="thumbSrc"
            class="max-h-[100svh] max-w-full object-contain lg:max-h-[90svh] lg:rounded-2xl lg:shadow-[var(--shadow-float)]"
            controls
            autoplay
            loop
            playsinline
            :muted="muted"
          />

          <!-- Subtle swipe hint at bottom edge (mobile only, fades after
               any swipe so we don't nag). -->
          <div
            v-if="(hasNext || hasPrev) && touchDeltaY === 0"
            class="pointer-events-none absolute bottom-[max(4.5rem,env(safe-area-inset-bottom))] left-1/2 -translate-x-1/2 text-center text-[11px] text-white/40 lg:hidden"
          >
            <div class="flex flex-col items-center gap-0.5">
              <svg
                class="h-4 w-4 opacity-70"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="1.8"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <path d="M12 5v14" />
                <path d="m7 14 5 5 5-5" />
              </svg>
              <span>swipe</span>
            </div>
          </div>
        </div>

        <!-- Desktop metadata sidebar: always visible, scrolls internally. -->
        <aside
          class="glass hidden shrink-0 overflow-y-auto rounded-bl-3xl px-7 py-8 text-[14px] text-ink-100 lg:block lg:max-w-[28rem]"
        >
          <Metadata
            :item="item"
            :copied="copied"
            :can-delete="canDelete"
            @copy="copy"
            @delete-clicked="confirmDelete"
          />
        </aside>
      </div>

      <!-- Mobile metadata bottom sheet. Slides up from the bottom edge on
           demand; backdrop-click dismisses. Not rendered on desktop —
           that layout uses the always-visible sidebar above. Marked
           swipe-ignore so the user can scroll the sheet freely without
           triggering navigation. -->
      <Transition name="sheet">
        <div
          v-if="sheetOpen"
          data-swipe-ignore
          class="absolute inset-0 z-30 flex flex-col justify-end lg:hidden"
          @click.self="sheetOpen = false"
        >
          <div
            class="glass max-h-[85svh] overflow-y-auto rounded-t-3xl px-5 pt-3 pb-[max(1.5rem,env(safe-area-inset-bottom))] text-[14px] text-ink-100 shadow-[0_-20px_60px_-20px_rgba(2,6,23,0.9)]"
          >
            <!-- Drag handle (visual affordance; we use the button for
                 keyboard accessibility). -->
            <div
              class="mx-auto mb-4 h-1 w-10 rounded-full bg-white/20"
              aria-hidden="true"
            />
            <Metadata
              :item="item"
              :copied="copied"
              :can-delete="canDelete"
              @copy="copy"
              @delete-clicked="confirmDelete"
            />
          </div>
        </div>
      </Transition>
    </div>
  </Transition>
</template>

<style scoped>
/*
 * Touch-action strategy
 * ---------------------
 * We disable Chrome's built-in touch gestures (pan/zoom/double-tap-zoom)
 * on the drawer root so its touchmove stream doesn't get swallowed while
 * the browser decides whether it's a scroll/zoom/tap. At `lg+` we restore
 * `auto` because desktop doesn't use swipe nav and the sidebar scrolls.
 *
 * Descendants marked `data-swipe-ignore` explicitly re-enable vertical
 * panning so the bottom-sheet metadata panel (and anything else with
 * overflow) can be scrolled by the user. The JS swipe handler filters
 * these same elements out on `touchstart`, so the browser's native scroll
 * and our custom swipe nav never fight.
 */
.drawer-root {
  touch-action: none;
  overscroll-behavior: contain;
}
.drawer-root [data-swipe-ignore] {
  touch-action: pan-y;
}
@media (min-width: 1024px) {
  .drawer-root,
  .drawer-root [data-swipe-ignore] {
    touch-action: auto;
  }
}

.sheet-enter-active,
.sheet-leave-active {
  transition:
    opacity 0.22s ease,
    transform 0.28s cubic-bezier(0.22, 1, 0.36, 1);
}
.sheet-enter-active > div,
.sheet-leave-active > div {
  transition: transform 0.28s cubic-bezier(0.22, 1, 0.36, 1);
}
.sheet-enter-from,
.sheet-leave-to {
  opacity: 0;
}
.sheet-enter-from > div,
.sheet-leave-to > div {
  transform: translateY(100%);
}
</style>
