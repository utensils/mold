<script setup lang="ts">
/*
 * Media tile — square print tile for library grids (spec §06). Rest state
 * carries the inset top highlight; hover lifts with a drop shadow. Optional
 * NEW badge for fresh prints; the "overlay" slot pins extra badges (e.g. a
 * video duration) to the bottom-right corner.
 *
 * Loading discipline: until the image bytes land (async blob thumbnails can
 * take a moment per tile) the tile shows a quiet shimmer, never the browser's
 * broken-image glyph with the prompt spelled out as alt text — a grid of
 * loading tiles must read as a grid, not a wall of paragraphs.
 */
import { ref, useSlots, watch } from "vue";

const props = withDefaults(
  defineProps<{
    src: string;
    alt: string;
    /** Marks a just-developed print with a NEW badge. */
    fresh?: boolean;
  }>(),
  { fresh: false },
);

const emit = defineEmits<{ open: [] }>();

const slots = useSlots();
const loaded = ref(false);
watch(
  () => props.src,
  () => {
    loaded.value = false;
  },
);
</script>

<template>
  <button
    type="button"
    class="ms-tile"
    :data-loaded="loaded"
    @click="emit('open')"
  >
    <span v-if="!loaded" class="ms-tile__ghost" aria-hidden="true" />
    <img
      v-if="src"
      class="ms-tile__img"
      :src="src"
      :alt="alt"
      loading="lazy"
      decoding="async"
      @load="loaded = true"
    />
    <span v-if="fresh" class="ms-tile__fresh">New</span>
    <span v-if="slots.overlay" class="ms-tile__overlay">
      <slot name="overlay" />
    </span>
  </button>
</template>

<style scoped>
.ms-tile {
  position: relative;
  display: block;
  width: 100%;
  aspect-ratio: 1;
  border: 0;
  padding: 0;
  border-radius: var(--mold-radius-2);
  overflow: hidden;
  background: var(--mold-media-bed);
  cursor: pointer;
  transition:
    transform var(--mold-dur-quick) var(--mold-ease-out),
    box-shadow var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-tile:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.4);
}

.ms-tile:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}

.ms-tile__img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  opacity: 1;
  transition: opacity var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-tile[data-loaded="false"] .ms-tile__img {
  /* Pending bytes: keep the element (so @load fires) but never its alt text
   * or the broken-image glyph. */
  opacity: 0;
}

.ms-tile__ghost {
  position: absolute;
  inset: 0;
  background-color: color-mix(
    in srgb,
    var(--mold-media-bed) 82%,
    var(--mold-blue) 18%
  );
  background-image:
    radial-gradient(
      circle at 28% 24%,
      color-mix(in srgb, var(--mold-sapphire) 16%, transparent),
      transparent 34%
    ),
    radial-gradient(
      circle at 72% 76%,
      color-mix(in srgb, var(--mold-blue) 14%, transparent),
      transparent 38%
    ),
    linear-gradient(
      100deg,
      transparent 20%,
      color-mix(in srgb, var(--mold-text) 9%, transparent) 50%,
      transparent 80%
    );
  background-size: 220% 100%;
  animation: ms-tile-shimmer 1.4s ease-in-out infinite;
}

.ms-tile__ghost::after {
  content: "Loading preview";
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  border: 1px solid color-mix(in srgb, var(--mold-text) 20%, transparent);
  border-radius: 999px;
  padding: 5px 9px;
  background: color-mix(in srgb, var(--mold-media-bed) 78%, transparent);
  color: var(--mold-text-dim);
  font-family: var(--mold-font-mono);
  font-size: 9px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  white-space: nowrap;
}

@media (prefers-reduced-motion: reduce) {
  .ms-tile__ghost {
    animation: none;
  }
}

@keyframes ms-tile-shimmer {
  0% {
    background-position: 130% 0;
  }
  100% {
    background-position: -90% 0;
  }
}

.ms-tile__fresh {
  position: absolute;
  top: 8px;
  left: 8px;
  font-family: var(--mold-font-mono);
  font-size: 8.5px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  background: var(--mold-blue);
  color: var(--mold-on-accent);
  padding: 2px 6px;
  border-radius: var(--mold-radius-1);
}

.ms-tile__overlay {
  position: absolute;
  bottom: 8px;
  right: 8px;
  display: flex;
  align-items: center;
  gap: 4px;
}
</style>
