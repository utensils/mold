<script setup lang="ts">
/*
 * Recent grid (Mold Studio Create) — the uniform print strip under the canvas.
 * The old Create page reused the Gallery's masonry feed here, which packed
 * variable-aspect cards into CSS columns: mp4 tiles came out half-height and
 * narrow, and the format/type/time badges stacked on top of each other. The
 * prototype's Recent block is a plain fixed grid of square tiles, so this
 * component renders one `MediaTile` per print — square, aspect-fit (cover)
 * thumbnails, a single non-overlapping video badge — long enough to fill a
 * large Create workspace, with a "view all" link into the gallery for the rest.
 */
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { RouterLink } from "vue-router";
import MediaTile from "@ui/components/MediaTile.vue";
import Icon from "@ui/components/Icon.vue";
import { thumbnailUrl } from "../../api";
import { mediaKind, type GalleryImage } from "../../types";

const props = withDefaults(
  defineProps<{
    entries: GalleryImage[];
    /** Cap the number of tiles rendered; the rest live in the gallery. */
    limit?: number;
    /**
     * Cap the grid to this many ROWS at the current breakpoint. Recent sits
     * under the sticky composer on Create, where an uncapped strip is the
     * whole page below the fold; two rows plus the "see all" link is the
     * mock's shape. Null means the `limit` alone decides.
     */
    maxRows?: number | null;
  }>(),
  { limit: 18, maxRows: null },
);

const emit = defineEmits<{
  open: [item: GalleryImage];
  "context-menu": [
    payload: {
      item: GalleryImage;
      x: number;
      y: number;
      trigger: HTMLElement | null;
    },
  ];
}>();

/** Columns the grid resolved at this width, read back from the layout so the
 * row cap is a slice (nothing past it in the DOM) rather than a clip. */
const gridEl = ref<HTMLElement | null>(null);
const columns = ref(0);
function measureColumns(): void {
  const el = gridEl.value;
  if (!el) return;
  const tracks = getComputedStyle(el).gridTemplateColumns;
  const count = tracks ? tracks.trim().split(/\s+/).filter(Boolean).length : 0;
  if (count > 0) columns.value = count;
}
/* The grid is a v-else branch: on a fresh page the entries arrive after
 * mount, so the element is watched rather than read once. */
let observer: ResizeObserver | null = null;
watch(
  gridEl,
  (el) => {
    observer?.disconnect();
    observer = null;
    if (!el) return;
    measureColumns();
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(measureColumns);
      observer.observe(el);
    }
  },
  { flush: "post" },
);
onBeforeUnmount(() => observer?.disconnect());

const cap = computed(() => {
  if (props.maxRows && props.maxRows > 0 && columns.value > 0) {
    return Math.min(props.limit, columns.value * props.maxRows);
  }
  return props.limit;
});
const shown = computed(() => props.entries.slice(0, cap.value));
const overflow = computed(() =>
  Math.max(0, props.entries.length - shown.value.length),
);

function isVideo(item: GalleryImage): boolean {
  return mediaKind(item.format, item.filename) === "video";
}
/** A mesh has no motion and no waveform — the gallery grid's own 3D badge is
 * the only thing that tells its square thumbnail apart from a still. */
function isMesh(item: GalleryImage): boolean {
  return mediaKind(item.format, item.filename) === "mesh";
}
function tileAlt(item: GalleryImage): string {
  return item.metadata.prompt || item.filename;
}
function openContextMenu(item: GalleryImage, event: MouseEvent): void {
  emit("context-menu", {
    item,
    x: event.clientX,
    y: event.clientY,
    trigger: event.currentTarget as HTMLElement | null,
  });
}
</script>

<template>
  <div
    class="recent"
    data-test="recent-grid"
    :data-max-rows="maxRows ?? undefined"
  >
    <div
      v-if="shown.length === 0"
      class="recent__empty"
      data-test="recent-empty"
    >
      no prints yet — your generations land here.
    </div>
    <div v-else ref="gridEl" class="recent__grid">
      <MediaTile
        v-for="item in shown"
        :key="item.filename"
        :src="thumbnailUrl(item.filename)"
        :alt="tileAlt(item)"
        :data-test="`recent-tile`"
        @open="emit('open', item)"
        @contextmenu.prevent.stop="openContextMenu(item, $event)"
      >
        <template v-if="isVideo(item)" #overlay>
          <span class="recent__badge" data-test="recent-video-badge">
            <Icon name="play" :size="11" />
            video
          </span>
        </template>
        <!-- Same mark as GalleryGrid's mesh tile so one print reads the same
             on both surfaces. -->
        <template v-else-if="isMesh(item)" #overlay>
          <span class="recent__badge" data-test="recent-mesh-badge">
            <svg
              class="recent__badge-glyph"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.8"
              stroke-linejoin="round"
              aria-hidden="true"
            >
              <path d="M12 2.6 20 7v10l-8 4.4L4 17V7z" />
              <path d="M4 7l8 4.4L20 7" />
              <path d="M12 11.4V21.4" />
            </svg>
            3D
          </span>
        </template>
      </MediaTile>
    </div>

    <RouterLink
      v-if="overflow > 0"
      to="/library"
      class="recent__more"
      data-test="recent-view-all"
    >
      See all {{ entries.length }} in My images
    </RouterLink>
  </div>
</template>

<style scoped>
.recent__grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
}

@media (min-width: 480px) {
  .recent__grid {
    grid-template-columns: repeat(3, 1fr);
  }
}
@media (min-width: 700px) {
  .recent__grid {
    grid-template-columns: repeat(4, 1fr);
  }
}
@media (min-width: 1100px) {
  .recent__grid {
    grid-template-columns: repeat(5, 1fr);
  }
}

.recent__badge {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  padding: 2px 7px 2px 6px;
  border-radius: var(--radius-pill);
  background: rgba(0, 0, 0, 0.62);
  color: #fff;
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  backdrop-filter: blur(4px);
}

.recent__badge-glyph {
  width: 11px;
  height: 11px;
}

.recent__empty {
  border: 1px solid var(--edge);
  border-radius: var(--radius-card);
  background: var(--bench);
  color: var(--ink-3);
  font-family: var(--f-mono);
  font-size: 12px;
  padding: 26px;
  text-align: center;
}

.recent__more {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-top: 12px;
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--ink-3);
  text-decoration: none;
}
.recent__more:hover {
  color: var(--safelight);
}
</style>
