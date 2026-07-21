<script setup lang="ts">
/*
 * Media tile — square print tile for library grids (spec §06). Rest state
 * carries the inset top highlight; hover lifts with a drop shadow. Optional
 * NEW badge for fresh prints; the "overlay" slot pins extra badges (e.g. a
 * video duration) to the bottom-right corner.
 */
import { useSlots } from "vue";

withDefaults(
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
</script>

<template>
  <button type="button" class="ms-tile" @click="emit('open')">
    <img class="ms-tile__img" :src="src" :alt="alt" loading="lazy" />
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
  border-radius: var(--radius-control);
  overflow: hidden;
  background: var(--print);
  cursor: pointer;
  box-shadow: inset 0 1px 0 var(--card-hi);
  transition:
    transform var(--dur-quick) var(--ease),
    box-shadow var(--dur-quick) var(--ease);
}

.ms-tile:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.4);
}

.ms-tile:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-tile__img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.ms-tile__fresh {
  position: absolute;
  top: 8px;
  left: 8px;
  font-family: var(--f-mono);
  font-size: 8.5px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  background: var(--safelight);
  color: var(--on-accent);
  padding: 2px 6px;
  border-radius: var(--radius-control-sm);
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
