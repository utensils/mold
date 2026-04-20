<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { imageUrl, thumbnailUrl } from "../api";
import type { GalleryImage } from "../types";
import { mediaKind } from "../types";
import Metadata from "./Metadata.vue";

/*
 * Full-viewport detail drawer. The visual chrome (nav buttons, counter,
 * aside) is driven by the new `.drawer-*` tokens so it adopts Studio/Lab
 * colours via `--accent` / `--accent-line`. Swipe-nav, keyboard shortcuts,
 * and touch-action carve-outs are preserved from the previous version.
 */

const props = defineProps<{
  item: GalleryImage | null;
  hasPrev: boolean;
  hasNext: boolean;
  index: number;
  total: number;
  canDelete: boolean;
  muted: boolean;
}>();

const emit = defineEmits<{
  (e: "close"): void;
  (e: "prev"): void;
  (e: "next"): void;
  (e: "delete", item: GalleryImage): void;
  (e: "rerun", item: GalleryImage): void;
}>();

const copied = ref<false | "prompt" | "seed">(false);
let copyTimer: ReturnType<typeof setTimeout> | null = null;

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

// ── Swipe navigation ──────────────────────────────────────────────────────

const SWIPE_THRESHOLD_PX = 60;
const SWIPE_MAX_HORIZONTAL_DRIFT = 80;

const touchStartY = ref<number | null>(null);
const touchStartX = ref<number | null>(null);
const touchDeltaY = ref(0);

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

  if (Math.abs(dx) > SWIPE_MAX_HORIZONTAL_DRIFT) return;

  if (dy > SWIPE_THRESHOLD_PX && props.hasNext) emit("next");
  else if (dy < -SWIPE_THRESHOLD_PX && props.hasPrev) emit("prev");
}

function onKey(e: KeyboardEvent) {
  if (!props.item) return;
  if (e.key === "Escape") {
    if (sheetOpen.value) sheetOpen.value = false;
    else emit("close");
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
    /* some contexts block clipboard */
  }
}

function confirmDelete() {
  if (!props.item) return;
  if (!window.confirm(`Delete ${props.item.filename}? This can't be undone.`))
    return;
  emit("delete", props.item);
}

function onRerun(item: GalleryImage) {
  emit("rerun", item);
}
</script>

<template>
  <Transition name="drawer">
    <div
      v-if="item"
      class="drawer drawer-root"
      @touchstart.passive="onTouchStart"
      @touchmove.passive="onTouchMove"
      @touchend.passive="onTouchEnd"
      @touchcancel.passive="onTouchEnd"
    >
      <!-- Top bar: close + counter + actions -->
      <div data-swipe-ignore class="drawer-topbar">
        <button
          class="drawer-iconbtn"
          aria-label="Close"
          @click="emit('close')"
        >
          <svg
            width="16"
            height="16"
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

        <div v-if="total > 0" class="drawer-counter">
          <button
            :disabled="!hasPrev"
            class="drawer-iconbtn"
            style="width: 30px; height: 30px"
            aria-label="Previous"
            @click="emit('prev')"
          >
            <svg
              width="14"
              height="14"
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
          <span>{{ index + 1 }} / {{ total }}</span>
          <button
            :disabled="!hasNext"
            class="drawer-iconbtn"
            style="width: 30px; height: 30px"
            aria-label="Next"
            @click="emit('next')"
          >
            <svg
              width="14"
              height="14"
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
        </div>

        <div class="drawer-topbar-actions">
          <a
            class="drawer-iconbtn"
            :href="mediaSrc"
            :download="item.filename"
            aria-label="Download"
            @click.stop
          >
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
              aria-hidden="true"
            >
              <path d="M12 3v13" />
              <path d="m6 12 6 6 6-6" />
              <path d="M5 21h14" />
            </svg>
          </a>
          <button
            class="drawer-iconbtn lg:hidden"
            :aria-pressed="sheetOpen"
            aria-label="Toggle details"
            style="display: grid"
            @click="sheetOpen = !sheetOpen"
          >
            <svg
              width="16"
              height="16"
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
      </div>

      <!-- Desktop prev/next floating rails -->
      <button
        v-if="hasPrev"
        data-swipe-ignore
        class="drawer-nav drawer-nav-prev hidden lg:grid"
        aria-label="Previous"
        @click="emit('prev')"
      >
        <svg
          width="20"
          height="20"
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
        class="drawer-nav drawer-nav-next hidden lg:grid"
        aria-label="Next"
        @click="emit('next')"
      >
        <svg
          width="20"
          height="20"
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

      <div class="drawer-body">
        <div
          class="drawer-media"
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
            class="drawer-img"
          />
          <video
            v-else
            data-swipe-ignore
            :src="mediaSrc"
            :poster="thumbSrc"
            class="drawer-video-el"
            controls
            autoplay
            loop
            playsinline
            :muted="muted"
          />
        </div>

        <aside class="drawer-aside hidden lg:block">
          <Metadata
            :item="item"
            :copied="copied"
            :can-delete="canDelete"
            @copy="copy"
            @rerun="onRerun"
            @delete-clicked="confirmDelete"
          />
        </aside>
      </div>

      <!-- Mobile bottom sheet -->
      <Transition name="sheet">
        <div
          v-if="sheetOpen"
          data-swipe-ignore
          class="absolute inset-0 z-30 flex flex-col justify-end lg:hidden"
          style="pointer-events: none"
          @click.self="sheetOpen = false"
        >
          <div
            class="glass drawer-sheet-body"
            style="
              pointer-events: auto;
              max-height: 85svh;
              overflow-y: auto;
              border-top-left-radius: 24px;
              border-top-right-radius: 24px;
              padding: 12px 20px calc(env(safe-area-inset-bottom, 0px) + 24px);
              box-shadow: 0 -20px 60px -20px rgba(0, 0, 0, 0.9);
            "
          >
            <div
              style="
                width: 40px;
                height: 4px;
                border-radius: 999px;
                background: color-mix(in oklab, var(--fg) 20%, transparent);
                margin: 4px auto 14px;
              "
              aria-hidden="true"
            />
            <Metadata
              :item="item"
              :copied="copied"
              :can-delete="canDelete"
              @copy="copy"
              @rerun="onRerun"
              @delete-clicked="confirmDelete"
            />
          </div>
        </div>
      </Transition>
    </div>
  </Transition>
</template>

<style scoped>
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
