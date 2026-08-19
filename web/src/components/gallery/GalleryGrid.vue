<script setup lang="ts">
/*
 * Gallery grid — the grid-first Mold Studio library view (spec §06, prototype
 * WEB GALLERY). Renders session prints as square @ui MediaTiles on a responsive
 * grid whose target pixel size comes from the shared Library toolbar. Freshly
 * arrived prints carry a NEW badge; video
 * prints show a play glyph + duration in the tile's overlay corner.
 *
 * Selection is gallery-only: when `selectMode` is on, a transparent hit layer
 * over each tile toggles its selection (Finder-style shift/meta), and an
 * empty-space marquee drag paints a selection rectangle. The tile grid is
 * chunked so a 1000+ print library doesn't mount every node up front.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import MediaTile from "@ui/components/MediaTile.vue";
import Icon from "@ui/components/Icon.vue";
import { printKey } from "../../lib/multiHostGallery";
import { useThumbnailSources } from "../../composables/useThumbnailSources";
import type { GalleryImage } from "../../types";
import type { ModelInfoExtended } from "../../types";
import { mediaKind } from "../../types";
import { modelDisplayNameForId } from "@studio/lib/modelDisplay";
import { purgeCountdownFromPurgeAt } from "@studio/lib/libraryOrganization";

const props = withDefaults(
  defineProps<{
    entries: GalleryImage[];
    models?: ModelInfoExtended[];
    loading: boolean;
    thumbnailSize?: number;
    selectMode?: boolean;
    /** Selected print keys — `hostId|filename`, never a bare filename. */
    selection?: Set<string>;
    /** Print keys that arrived this session — badged NEW. */
    fresh?: Set<string>;
    /** Trash scope: tiles wear a purge countdown and hover Restore /
     * Delete forever actions instead of opening on click alone. */
    trash?: boolean;
    /** Wall clock for the purge countdown (injectable for tests). */
    now?: number;
  }>(),
  {
    selectMode: false,
    selection: () => new Set<string>(),
    fresh: () => new Set<string>(),
    models: () => [],
    thumbnailSize: 220,
    trash: false,
    now: undefined,
  },
);
const modelLabel = (name: string) => modelDisplayNameForId(name, props.models);

const emit = defineEmits<{
  (e: "open", item: GalleryImage): void;
  (
    e: "toggle-select",
    payload: { item: GalleryImage; shift: boolean; meta: boolean },
  ): void;
  (e: "drag-select", payload: { keys: string[] }): void;
  (
    e: "context-menu",
    payload: { item: GalleryImage; x: number; y: number },
  ): void;
  (e: "restore", item: GalleryImage): void;
  (e: "delete-forever", item: GalleryImage): void;
}>();

function purgeLabel(entry: GalleryImage): string {
  return purgeCountdownFromPurgeAt(entry.purge_at, props.now ?? Date.now())
    .label;
}
function purgeKind(entry: GalleryImage): string {
  return purgeCountdownFromPurgeAt(entry.purge_at, props.now ?? Date.now())
    .kind;
}

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
function isAudio(entry: GalleryImage): boolean {
  return tileKind(entry) === "audio";
}
// The grid always shows the cached thumbnail (fast, poster-friendly for video),
// addressed on the host that owns the print (see useThumbnailSources).
const { srcFor: tileSrc } = useThumbnailSources();

function hostLabel(entry: GalleryImage): string {
  return (entry as { hostLabel?: string }).hostLabel ?? "";
}

/** Composite identity for a tile: two hosts can hold the same filename. */
function keyOf(entry: GalleryImage): string {
  return printKey(entry as { hostId?: string; filename: string });
}

/** Hover strip: `title · model · S seed` — the title leads when one exists. */
function stripLabel(entry: GalleryImage): string {
  const base = `${modelLabel(entry.metadata.model)} · S ${entry.metadata.seed}`;
  const title = entry.title?.trim();
  return title ? `${title} · ${base}` : base;
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

function onContextMenu(entry: GalleryImage, event: MouseEvent) {
  emit("context-menu", { item: entry, x: event.clientX, y: event.clientY });
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
  if (target?.closest("[data-print-key]")) return;
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
  emit("drag-select", { keys: Array.from(final) });
}

function onPointerUp() {
  window.removeEventListener("pointermove", onPointerMove);
  drag = null;
  dragBox.value = null;
}

function collectHits(x: number, y: number, w: number, h: number): string[] {
  if (!gridRoot.value) return [];
  const cells =
    gridRoot.value.querySelectorAll<HTMLElement>("[data-print-key]");
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
    const key = cell.dataset.printKey;
    if (key) hits.push(key);
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
    :style="{ '--gallery-thumbnail-size': `${thumbnailSize}px` }"
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
          :key="keyOf(entry)"
          class="gg__cell"
          :data-filename="entry.filename"
          :data-print-key="keyOf(entry)"
          :data-selected="selection.has(keyOf(entry)) ? 'true' : 'false'"
          @contextmenu.prevent="onContextMenu(entry, $event)"
        >
          <MediaTile
            :src="tileSrc(entry)"
            :alt="entry.metadata.prompt || entry.filename"
            :fresh="fresh.has(keyOf(entry))"
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
            <template v-else-if="isAudio(entry)" #overlay>
              <span class="gg__vbadge">
                <svg
                  class="gg__vplay"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  aria-hidden="true"
                >
                  <path d="M12 3v10.55A4 4 0 1 0 14 17V7h4V3h-6z" />
                </svg>
                <span v-if="durationLabel(entry)">{{
                  durationLabel(entry)
                }}</span>
              </span>
            </template>
          </MediaTile>

          <span
            v-if="entry.favorite"
            class="gg__fav"
            data-test="favorite-badge"
            title="Favorite"
            aria-label="Favorite"
          >
            <Icon name="heart" :size="13" :stroke-width="2" />
          </span>

          <span
            v-if="trash"
            class="gg__purge"
            :data-kind="purgeKind(entry)"
            data-test="purge-chip"
          >
            {{ purgeLabel(entry) }}
          </span>

          <span
            v-if="hostLabel(entry)"
            class="gg__host"
            data-test="host-badge"
            :title="`Generated on ${hostLabel(entry)}`"
          >
            {{ hostLabel(entry) }}
          </span>
          <span
            v-if="trash && !selectMode"
            class="gg__trash-actions"
            data-test="trash-actions"
          >
            <button
              type="button"
              class="gg__ta"
              data-test="tile-restore"
              @click.stop="emit('restore', entry)"
            >
              Restore
            </button>
            <button
              type="button"
              class="gg__ta gg__ta--danger"
              data-test="tile-delete-forever"
              @click.stop="emit('delete-forever', entry)"
            >
              Delete forever
            </button>
          </span>
          <span
            v-else
            class="gg__metadata"
            data-test="print-metadata"
            :title="stripLabel(entry)"
          >
            {{ stripLabel(entry) }}
          </span>

          <!-- Selection hit layer (select mode only). Sits above the tile so a
               click toggles instead of opening; keeps shift/meta range logic. -->
          <button
            v-if="selectMode"
            type="button"
            class="gg__hit"
            :class="{ 'gg__hit--on': selection.has(keyOf(entry)) }"
            :aria-pressed="selection.has(keyOf(entry))"
            :aria-label="`${selection.has(keyOf(entry)) ? 'Deselect' : 'Select'} ${entry.filename}`"
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
  width: 100%;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
@media (min-width: 640px) {
  .gg__grid {
    grid-template-columns: repeat(
      auto-fill,
      minmax(min(var(--gallery-thumbnail-size), 100%), 1fr)
    );
  }
}

.gg__cell {
  position: relative;
  min-width: 0;
  aspect-ratio: 1;
  overflow: hidden;
  border-radius: var(--radius-control);
  contain: inline-size layout paint;
}

.gg__skel {
  aspect-ratio: 1;
  border-radius: var(--radius-control-lg);
  background: color-mix(in srgb, var(--rebate) 5%, transparent);
}

/* Video/animated badge stays top-right so gallery metadata owns the bottom edge. */
:deep(.ms-tile__overlay) {
  top: 8px;
  right: 8px;
  bottom: auto;
}

/* The cell deliberately clips intrinsic media. Keep keyboard focus inside that
 * paint boundary so the shared tile's outward ring remains fully visible. */
:deep(.ms-tile:focus-visible) {
  outline-offset: -2px;
}

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

.gg__host,
.gg__metadata {
  position: absolute;
  z-index: 1;
  pointer-events: none;
  color: var(--on-media);
  font-family: var(--f-mono);
  font-size: 10px;
  line-height: 1.4;
}

.gg__host {
  bottom: 8px;
  left: 8px;
  max-width: 70%;
  overflow: hidden;
  padding: 2px 6px;
  border-radius: var(--radius-control-sm);
  background: rgba(0, 0, 0, 0.62);
  text-overflow: ellipsis;
  white-space: nowrap;
  transition: opacity var(--dur-quick) var(--ease);
}

.gg__metadata {
  right: 0;
  bottom: 0;
  left: 0;
  overflow: hidden;
  padding: 3px 7px;
  background: rgba(0, 0, 0, 0.62);
  text-align: left;
  text-overflow: ellipsis;
  white-space: nowrap;
  transform: translateY(100%);
  transition: transform var(--dur-quick) var(--ease);
}

.gg__cell:hover .gg__host,
.gg__cell:focus-within .gg__host {
  opacity: 0;
}

.gg__cell:hover .gg__metadata,
.gg__cell:focus-within .gg__metadata {
  transform: translateY(0);
}

/* Favorite heart — bottom-right, filled with the accent; never color-only
 * since the glyph itself carries the meaning. */
.gg__fav {
  position: absolute;
  right: 8px;
  bottom: 8px;
  z-index: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.55);
  color: var(--safelight);
  pointer-events: none;
}
.gg__fav :deep(svg) {
  fill: currentColor;
}
.gg__cell:hover .gg__fav,
.gg__cell:focus-within .gg__fav {
  bottom: 30px;
}

/* Trash: purge countdown (top-left; the NEW slot is unused there) with the
 * warning tone on the chip only. */
.gg__purge {
  position: absolute;
  top: 8px;
  left: 8px;
  z-index: 1;
  max-width: calc(100% - 16px);
  padding: 2px 7px;
  border-radius: var(--radius-pill);
  background: rgba(0, 0, 0, 0.62);
  color: var(--on-media);
  font-family: var(--f-mono);
  font-size: 10px;
  font-weight: 700;
  line-height: 1.4;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  pointer-events: none;
}
.gg__purge[data-kind="purges"],
.gg__purge[data-kind="today"] {
  color: var(--warning);
}

/* Restore / Delete forever slide up on hover like the metadata strip. */
.gg__trash-actions {
  position: absolute;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 2;
  display: flex;
  gap: 6px;
  padding: 6px 7px;
  background: rgba(0, 0, 0, 0.62);
  transform: translateY(100%);
  transition: transform var(--dur-quick) var(--ease);
}
.gg__cell:hover .gg__trash-actions,
.gg__cell:focus-within .gg__trash-actions {
  transform: translateY(0);
}
.gg__ta {
  flex: 1;
  min-height: 28px;
  border: 1px solid rgba(255, 255, 255, 0.35);
  border-radius: var(--radius-control-sm);
  background: rgba(255, 255, 255, 0.12);
  color: var(--on-media);
  font-family: var(--f-body);
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
}
.gg__ta:hover {
  background: rgba(255, 255, 255, 0.22);
}
.gg__ta--danger {
  color: var(--stop);
  border-color: color-mix(in srgb, var(--stop) 60%, transparent);
}
.gg__ta:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: -2px;
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
  outline-offset: -2px;
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
