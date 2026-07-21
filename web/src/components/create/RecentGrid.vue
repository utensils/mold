<script setup lang="ts">
/*
 * Recent grid (Mold Studio Create) — the uniform print strip under the canvas.
 * The old Create page reused the Gallery's masonry feed here, which packed
 * variable-aspect cards into CSS columns: mp4 tiles came out half-height and
 * narrow, and the format/type/time badges stacked on top of each other. The
 * prototype's Recent block is a plain fixed grid of square tiles, so this
 * component renders one `MediaTile` per print — square, aspect-fit (cover)
 * thumbnails, a single non-overlapping video badge — capped to a handful with
 * a "view all" link into the gallery for the rest.
 */
import { computed } from "vue";
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
  }>(),
  { limit: 18 },
);

const emit = defineEmits<{ open: [item: GalleryImage] }>();

const shown = computed(() => props.entries.slice(0, props.limit));
const overflow = computed(() =>
  Math.max(0, props.entries.length - shown.value.length),
);

function isVideo(item: GalleryImage): boolean {
  return mediaKind(item.format, item.filename) === "video";
}
function tileAlt(item: GalleryImage): string {
  return item.metadata.prompt || item.filename;
}
</script>

<template>
  <div class="recent" data-test="recent-grid">
    <div
      v-if="shown.length === 0"
      class="recent__empty"
      data-test="recent-empty"
    >
      no prints yet — your generations land here.
    </div>
    <div v-else class="recent__grid">
      <MediaTile
        v-for="item in shown"
        :key="item.filename"
        :src="thumbnailUrl(item.filename)"
        :alt="tileAlt(item)"
        :data-test="`recent-tile`"
        @open="emit('open', item)"
      >
        <template v-if="isVideo(item)" #overlay>
          <span class="recent__badge" data-test="recent-video-badge">
            <Icon name="play" :size="11" />
            video
          </span>
        </template>
      </MediaTile>
    </div>

    <RouterLink
      v-if="overflow > 0"
      to="/"
      class="recent__more"
      data-test="recent-view-all"
    >
      view all {{ entries.length }} in gallery
      <Icon name="chevron-right" :size="13" />
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
