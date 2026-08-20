<script setup lang="ts">
/*
 * Library filter-chip row (V3 "Shelf"): ♥ Favorites toggle, the most-used tag
 * chips with counts (the rest behind "More tags…"), and the host chips that
 * used to live in the header. Purely presentational — the page owns the
 * filter state and its URL sync. Below 640px the row scrolls horizontally.
 */
import { computed, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import Popover from "@ui/components/Popover.vue";
import { tagKey } from "@studio/lib/libraryOrganization";
import type { TagCount } from "../../types";

/** Tag chips shown inline; the rest fold into "More tags…". */
const VISIBLE_TAG_CHIPS = 8;

export interface HostChipOption {
  id: string;
  label: string;
}

const props = withDefaults(
  defineProps<{
    /** Whether any host supports favorites/tags (else only host chips show). */
    organize: boolean;
    favoritesOnly: boolean;
    favoriteCount: number;
    tags: readonly TagCount[];
    activeTags: readonly string[];
    hostOptions: readonly HostChipOption[];
    hostFilter: string;
  }>(),
  {},
);

const emit = defineEmits<{
  (e: "toggle-favorites"): void;
  (e: "toggle-tag", tag: string): void;
  (e: "set-host", id: string): void;
}>();

const moreOpen = ref(false);

const activeKeys = computed(
  () => new Set(props.activeTags.map((tag) => tagKey(tag))),
);
const isActive = (tag: string) => activeKeys.value.has(tagKey(tag));

/** Active tags always stay visible, even when they rank past the cap. */
const visibleTags = computed(() => {
  const head = props.tags.slice(0, VISIBLE_TAG_CHIPS);
  const pinned = props.tags
    .slice(VISIBLE_TAG_CHIPS)
    .filter((tag) => isActive(tag.name));
  return [...head, ...pinned];
});
const overflowTags = computed(() =>
  props.tags.slice(VISIBLE_TAG_CHIPS).filter((tag) => !isActive(tag.name)),
);
const showRow = computed(() => props.organize || props.hostOptions.length > 1);
</script>

<template>
  <div v-if="showRow" class="lcr" data-test="library-chip-row">
    <template v-if="organize">
      <button
        type="button"
        class="lcr__chip lcr__chip--fav"
        :data-on="favoritesOnly ? 'true' : undefined"
        :aria-pressed="favoritesOnly"
        data-test="chip-favorites"
        @click="emit('toggle-favorites')"
      >
        <Icon name="heart" :size="12" :stroke-width="2" class="lcr__heart" />
        Favorites
        <span class="lcr__n">{{ favoriteCount }}</span>
      </button>
      <span v-if="tags.length > 0" class="lcr__vr" aria-hidden="true"></span>
      <button
        v-for="tag in visibleTags"
        :key="tag.name"
        type="button"
        class="lcr__chip"
        :data-on="isActive(tag.name) ? 'true' : undefined"
        :aria-pressed="isActive(tag.name)"
        data-test="chip-tag"
        @click="emit('toggle-tag', tag.name)"
      >
        <Icon name="tag" :size="12" />
        {{ tag.name }}
        <span class="lcr__n">{{ tag.count }}</span>
      </button>
      <Popover
        v-if="overflowTags.length > 0"
        :open="moreOpen"
        label="More tags"
        @update:open="moreOpen = $event"
      >
        <template #trigger>
          <button
            type="button"
            class="lcr__chip lcr__chip--more"
            :aria-expanded="moreOpen"
            data-test="chip-more-tags"
            @click="moreOpen = !moreOpen"
          >
            More tags…
            <span class="lcr__n">+{{ overflowTags.length }}</span>
          </button>
        </template>
        <div class="lcr__more" data-test="more-tags-panel">
          <button
            v-for="tag in overflowTags"
            :key="tag.name"
            type="button"
            class="lcr__chip"
            data-test="chip-tag"
            @click="
              emit('toggle-tag', tag.name);
              moreOpen = false;
            "
          >
            <Icon name="tag" :size="12" />
            {{ tag.name }}
            <span class="lcr__n">{{ tag.count }}</span>
          </button>
        </div>
      </Popover>
    </template>

    <span class="lcr__flex"></span>

    <div
      v-if="hostOptions.length > 1"
      class="lcr__hosts"
      role="group"
      aria-label="Filter by machine"
    >
      <button
        v-for="option in [{ id: 'all', label: 'All machines' }, ...hostOptions]"
        :key="option.id"
        type="button"
        class="lcr__host"
        :data-on="hostFilter === option.id ? 'true' : undefined"
        data-test="gallery-host-filter"
        @click="emit('set-host', option.id)"
      >
        {{ option.label }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.lcr {
  display: flex;
  align-items: center;
  gap: 6px;
  min-height: 40px;
  margin: -6px 0 14px;
  overflow-x: auto;
  scrollbar-width: none;
}
.lcr::-webkit-scrollbar {
  display: none;
}
.lcr__flex {
  flex: 1;
  min-width: 8px;
}
.lcr__chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  flex: 0 0 auto;
  height: 30px;
  padding: 0 10px 0 9px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-pill);
  background: transparent;
  color: var(--ink-2);
  font-family: var(--f-body);
  font-size: 12px;
  white-space: nowrap;
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease),
    background var(--dur-quick) var(--ease);
}
.lcr__chip:hover {
  border-color: var(--ink-3);
  color: var(--rebate);
}
.lcr__chip[data-on="true"] {
  border-color: var(--sel-border);
  color: var(--sel-ink);
  background: var(--sel-bg);
  box-shadow: var(--sel-ring);
}
.lcr__chip:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
.lcr__chip--fav[data-on="true"] .lcr__heart :deep(svg),
.lcr__chip--fav[data-on="true"] :deep(svg) {
  fill: currentColor;
}
.lcr__chip--more {
  color: var(--ink-3);
}
.lcr__n {
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
}
.lcr__chip[data-on="true"] .lcr__n {
  color: var(--sel-ink);
}
.lcr__vr {
  flex: 0 0 1px;
  width: 1px;
  height: 18px;
  margin: 0 2px;
  background: var(--edge);
}
.lcr__more {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  width: min(360px, calc(100vw - 32px));
  padding: 10px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control-lg);
  background: var(--bench);
  box-shadow: var(--shadow-raised);
}
.lcr__hosts {
  display: inline-flex;
  gap: 6px;
  flex: 0 0 auto;
}
.lcr__host {
  min-height: 30px;
  padding: 0 10px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-pill);
  background: var(--bath);
  color: var(--ink-2);
  font-family: var(--f-mono);
  font-size: 10px;
  white-space: nowrap;
  cursor: pointer;
}
.lcr__host[data-on="true"] {
  border-color: var(--safelight);
  color: var(--rebate);
  background: var(--sel-bg);
}
.lcr__host:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
</style>
