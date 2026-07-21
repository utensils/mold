<script setup lang="ts">
/*
 * Gallery grid — the grid-first Mold Studio library view (spec §06, prototype
 * WEB GALLERY). Renders session prints as square @ui MediaTiles on a 2/3/4/5
 * column responsive grid. Freshly-arrived prints carry a NEW badge; video
 * prints show a play glyph + duration in the tile's overlay corner.
 *
 * Selection is gallery-only: when `selectMode` is on, a transparent hit layer
 * over each tile toggles its selection (Finder-style shift/meta), and an
 * empty-space marquee drag paints a selection rectangle. The tile grid is
 * chunked so a 1000+ print library doesn't mount every node up front.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import MediaTile from "@ui/components/MediaTile.vue";
import { thumbnailUrl } from "../../api";
import type { GalleryImage } from "../../types";
import { mediaKind } from "../../types";

const props = withDefaults(
  defineProps<{
    entries: GalleryImage[];
    loading: boolean;
    selectMode?: boolean;
    selection?: Set<string>;
    /** Filenames that arrived this session — badged NEW. */
    fresh?: Set<string>;
  }>(),
  {
    selectMode: false,
    selection: () => new Set<string>(),
    fresh: () => new Set<string>(),
  },
);

const emit = defineEmits<{
  (e: "open", item: GalleryImage): void;
  (
    e: "toggle-select",
    payload: { item: GalleryImage; shift: boolean; meta: boolean },
  ): void;
  (e: "drag-select", payload: { filenames: string[] }): void;
}>();

// ── Chunked rendering ──────────────────────────────────────────────────────
// Grid tiles pack densely so we render a large-ish chunk and grow it as the
// sentinel scrolls into view.
const PAGE_SIZE = 150;
const visibleCount = ref(PAGE_SIZE);
const sentinel = ref<HTMLElement | null>(null);

const visibleEntries = computed(() =>
  props.entries.slice(0, visibleCount.value),
);
const hasMore = computed(() => visibleCount.value < props.entries.length);

function loadMore() {
  visibleCount.value = Math.min(
    visibleCount.value + PAGE_SIZE,
    props.entries.length,
  );
}

let observer: IntersectionObserver | null = null;

function installObserver() {
  observer?.disconnect();
  if (!sentinel.value || typeof IntersectionObserver === "undefined") return;
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

onMounted(installObserver);
onBeforeUnmount(() => observer?.disconnect());

// Narrowing the filtered set (search / filter) should snap the window back to
// page one rather than keep hundreds of tiles mounted.
watch(
  () => props.entries,
  () => {
    visibleCount.value = Math.min(PAGE_SIZE, props.entries.length);
    queueMicrotask(installObserver);
  },
);

const skeletons = computed(() =>
  props.loading && props.entries.length === 0 ? 10 : 0,
);

// ── Tile helpers ───────────────────────────────────────────────────────────
function tileKind(entry: GalleryImage) {
  return mediaKind(entry.format, entry.filename);
}
function isMotion(entry: GalleryImage): boolean {
  const k = tileKind(entry);
  return k === "video" || k === "animated";
}
// The grid always shows the cached thumbnail (fast, poster-friendly for video).
function tileSrc(entry: GalleryImage): string {
  return thumbnailUrl(entry.filename);
}
function durationLabel(entry: GalleryImage): string {
  const frames = entry.metadata.frames;
  const fps = entry.metadata.fps;
  if (!frames || !fps || fps <= 0) return "";
  const secs = frames / fps;
  if (secs >= 60) {
    const m = Math.floor(secs / 60);
    const s = Math.round(secs % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  }
  return `${secs < 10 ? secs.toFixed(1) : Math.round(secs)}s`;
}

function onTileOpen(entry: GalleryImage) {
  // In select mode the hit layer intercepts clicks; this only fires from the
  // media tile itself (normal mode, or keyboard Enter/Space).
  if (props.selectMode) {
    emit("toggle-select", { item: entry, shift: false, meta: false });
    return;
  }
  emit("open", entry);
}

function onSelectClick(entry: GalleryImage, evt: MouseEvent) {
  emit("toggle-select", {
    item: entry,
    shift: evt.shiftKey,
    meta: evt.metaKey || evt.ctrlKey,
  });
}

// ── Marquee / drag selection ───────────────────────────────────────────────
const gridRoot = ref<HTMLElement | null>(null);
const dragBox = ref<{ x: number; y: number; w: number; h: number } | null>(
  null,
);

type DragState = {
  startX: number;
  startY: number;
  additive: boolean;
  started: boolean;
  base: Set<string>;
};
let drag: DragState | null = null;

function onPointerDown(evt: PointerEvent) {
  if (!props.selectMode || evt.button !== 0) return;
  const target = evt.target as HTMLElement | null;
  // Clicks on a tile / control toggle selection directly — only empty gaps
  // start a marquee.
  if (target?.closest("[data-filename]")) return;
  if (target?.closest("button, a, input, textarea")) return;
  drag = {
    startX: evt.clientX,
    startY: evt.clientY,
    additive: evt.shiftKey || evt.metaKey || evt.ctrlKey,
    started: false,
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
  if (!gridRoot.value) return [];
  const cells = gridRoot.value.querySelectorAll<HTMLElement>("[data-filename]");
  const hits: string[] = [];
  const right = x + w;
  const bottom = y + h;
  for (const cell of cells) {
    const rect = cell.getBoundingClientRect();
    if (
      rect.right < x ||
      rect.left > right ||
      rect.bottom < y ||
      rect.top > bottom
    ) {
      continue;
    }
    const name = cell.dataset.filename;
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
    ref="gridRoot"
    class="gg"
    :class="{ 'gg--selecting': selectMode }"
    @pointerdown="onPointerDown"
  >
    <!-- Loading skeletons -->
    <div v-if="skeletons > 0" class="gg__grid">
      <div
        v-for="i in skeletons"
        :key="`skel-${i}`"
        class="gg__skel ms-shimmer"
      ></div>
    </div>

    <template v-else>
      <div class="gg__grid">
        <div
          v-for="entry in visibleEntries"
          :key="entry.filename"
          class="gg__cell"
          :data-filename="entry.filename"
          :data-selected="selection.has(entry.filename) ? 'true' : 'false'"
        >
          <MediaTile
            :src="tileSrc(entry)"
            :alt="entry.metadata.prompt || entry.filename"
            :fresh="fresh.has(entry.filename)"
            @open="onTileOpen(entry)"
          >
            <template v-if="isMotion(entry)" #overlay>
              <span class="gg__vbadge">
                <svg
                  class="gg__vplay"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  aria-hidden="true"
                >
                  <path d="M8 5v14l11-7z" />
                </svg>
                <span v-if="durationLabel(entry)">{{
                  durationLabel(entry)
                }}</span>
              </span>
            </template>
          </MediaTile>

          <!-- Selection hit layer (select mode only). Sits above the tile so a
               click toggles instead of opening; keeps shift/meta range logic. -->
          <button
            v-if="selectMode"
            type="button"
            class="gg__hit"
            :class="{ 'gg__hit--on': selection.has(entry.filename) }"
            :aria-pressed="selection.has(entry.filename)"
            :aria-label="`${selection.has(entry.filename) ? 'Deselect' : 'Select'} ${entry.filename}`"
            @click="onSelectClick(entry, $event)"
          >
            <span class="gg__check" aria-hidden="true">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                stroke-width="3"
                stroke-linecap="round"
                stroke-linejoin="round"
              >
                <path d="m5 12 5 5L20 7" />
              </svg>
            </span>
          </button>
        </div>
      </div>

      <div
        v-if="entries.length > 0"
        ref="sentinel"
        class="gg__sentinel"
        aria-hidden="true"
      >
        <span v-if="hasMore">
          Loading more… ({{ visibleCount }}/{{ entries.length }})
        </span>
        <span v-else class="gg__sentinel-end">
          {{ entries.length }} prints
        </span>
      </div>
    </template>

    <!-- Marquee rectangle (viewport-fixed; pointer coords are viewport-relative). -->
    <div
      v-if="dragBox"
      class="gg__marquee"
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

<style scoped>
.gg {
  position: relative;
}
.gg--selecting {
  user-select: none;
}

.gg__grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}
@media (min-width: 640px) {
  .gg__grid {
    grid-template-columns: repeat(3, 1fr);
  }
}
@media (min-width: 900px) {
  .gg__grid {
    grid-template-columns: repeat(4, 1fr);
  }
}
@media (min-width: 1200px) {
  .gg__grid {
    grid-template-columns: repeat(5, 1fr);
  }
}

.gg__cell {
  position: relative;
}

.gg__skel {
  aspect-ratio: 1;
  border-radius: var(--radius-control-lg);
  background: color-mix(in srgb, var(--rebate) 5%, transparent);
}

/* Video/animated badge pinned bottom-right via MediaTile's overlay slot. */
.gg__vbadge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 7px 2px 5px;
  border-radius: var(--radius-pill);
  background: rgba(0, 0, 0, 0.62);
  color: var(--on-media);
  font-family: var(--f-mono);
  font-size: 10px;
  font-weight: 700;
  line-height: 1.4;
}
.gg__vplay {
  width: 10px;
  height: 10px;
}

.gg__hit {
  position: absolute;
  inset: 0;
  border: 2px solid transparent;
  border-radius: var(--radius-control-lg);
  background: transparent;
  cursor: pointer;
  transition: border-color var(--dur-quick) var(--ease);
}
.gg__hit--on {
  border-color: var(--safelight);
  background: var(--sel-bg);
}
.gg__hit:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.gg__check {
  position: absolute;
  top: 8px;
  right: 8px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  border: 2px solid rgba(255, 255, 255, 0.6);
  background: rgba(0, 0, 0, 0.4);
  color: transparent;
}
.gg__check svg {
  width: 14px;
  height: 14px;
}
.gg__hit--on .gg__check {
  border-color: var(--safelight);
  background: var(--safelight);
  color: var(--on-accent);
}

.gg__sentinel {
  margin-top: 22px;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 32px;
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-3);
}
.gg__sentinel-end {
  opacity: 0.7;
}

.gg__marquee {
  position: fixed;
  z-index: 40;
  border: 1px solid color-mix(in srgb, var(--safelight) 70%, transparent);
  background: var(--sel-bg);
  border-radius: var(--radius-control-sm);
  pointer-events: none;
}
</style>
