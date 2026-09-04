<script setup lang="ts">
/*
 * LibraryChipRow — the 30px filter strip under the Library header, in every
 * scope: the top tags (mono counts; active = AND), the "＋ tag" chip that
 * opens the searchable list of the rest, the open album as a removable chip,
 * and — after a dim "Made on" caption — one bordered mono chip per machine.
 * Favourites is a SCOPE, not a chip. With nothing filtered the row is a quiet
 * status line; with anything active it grows a **Clear filters** link.
 * Pure props/emits: the store owns every filter.
 */
import { computed, ref } from "vue";
import Chip from "@ui/components/Chip.vue";
import Icon from "@ui/components/Icon.vue";
import Popover from "@ui/components/Popover.vue";
import { tagKey } from "@studio/lib/libraryOrganization";
import type { TagCount } from "@studio/lib/api/galleryOrganization";

/** Tags shown inline before they fold into the "＋ tag" popover. */
const INLINE_TAG_LIMIT = 8;

interface HostChip {
  key: string;
  label: string;
  count: number;
}

const props = withDefaults(
  defineProps<{
    /** Hide the tags when no connected host can organize. */
    organize: boolean;
    /** Host-merged tags, already sorted (count desc, name). */
    tags: readonly TagCount[];
    /** Active tag filter (names). */
    activeTags: readonly string[];
    hostChips: readonly HostChip[];
    hostFilter: string;
    /** Open collection (drill-in) shown as a removable chip. */
    collectionName?: string | null;
  }>(),
  { collectionName: null },
);

const emit = defineEmits<{
  toggleTag: [name: string];
  "update:hostFilter": [key: string];
  clearFilters: [];
  exitCollection: [];
}>();

const activeKeys = computed(() => new Set(props.activeTags.map(tagKey)));
const isActive = (name: string) => activeKeys.value.has(tagKey(name));

/** Inline chips: the top N plus any active tag that would otherwise hide
 *  inside the "＋ tag" popover — an active filter must stay removable. */
const inlineTags = computed(() => {
  const top = props.tags.slice(0, INLINE_TAG_LIMIT);
  const shown = new Set(top.map((t) => tagKey(t.name)));
  const extras = props.tags.filter((t) => !shown.has(tagKey(t.name)) && isActive(t.name));
  return [...top, ...extras];
});
const hiddenCount = computed(() => Math.max(0, props.tags.length - inlineTags.value.length));

const moreOpen = ref(false);
const moreQuery = ref("");
const moreMatches = computed(() => {
  const q = tagKey(moreQuery.value);
  return props.tags.filter((t) => q === "" || tagKey(t.name).includes(q));
});

const anyFilter = computed(
  () => props.activeTags.length > 0 || props.hostFilter !== "all" || !!props.collectionName,
);

function closeMore() {
  moreOpen.value = false;
  moreQuery.value = "";
}
defineExpose({ closeMore, isOpen: () => moreOpen.value });
</script>

<template>
  <div
    class="flex shrink-0 items-center gap-1.5 overflow-x-auto border-b border-border bg-chrome px-3.5 py-2"
    data-test="library-chip-row"
    role="group"
    aria-label="Library filters"
  >
    <template v-if="collectionName">
      <button
        type="button"
        class="ms-lib-chip ms-lib-chip--on"
        data-test="collection-chip"
        :title="`Leave ${collectionName}`"
        @click="emit('exitCollection')"
      >
        <Icon name="collection" :size="12" />
        <span class="max-w-48 truncate">{{ collectionName }}</span>
        <span aria-hidden="true">×</span>
      </button>
      <span class="ms-lib-vr" aria-hidden="true" />
    </template>

    <template v-if="organize">
      <Chip
        v-for="tag in inlineTags"
        :key="tag.name"
        class="ms-lib-chip"
        :active="isActive(tag.name)"
        data-test="tag-chip"
        :data-tag="tag.name"
        @click="emit('toggleTag', tag.name)"
      >
        <Icon name="tag" :size="11" class="text-fg-dim" />
        <span class="max-w-32 truncate">{{ tag.name }}</span>
        <span class="ms-lib-chip__n">{{ tag.count }}</span>
      </Chip>
      <span v-if="tags.length === 0" class="text-micro text-fg-dim" data-test="no-tags">
        No tags yet
      </span>
      <Popover
        v-if="tags.length > 0"
        :open="moreOpen"
        label="All tags"
        @update:open="moreOpen = $event"
      >
        <template #trigger>
          <button
            type="button"
            class="ms-lib-chip ms-lib-chip--more"
            data-test="more-tags"
            aria-label="All tags"
            :aria-expanded="moreOpen"
            @click="moreOpen ? closeMore() : (moreOpen = true)"
          >
            ＋ tag
            <span v-if="hiddenCount > 0" class="ms-lib-chip__n">+{{ hiddenCount }}</span>
          </button>
        </template>
        <div class="flex w-60 flex-col gap-1.5" data-test="more-tags-panel">
          <label
            class="border-border-control flex h-7 items-center gap-1.5 rounded-control border bg-bg-deep px-2"
          >
            <Icon name="search" :size="12" class="shrink-0 text-fg-dim" />
            <input
              v-model="moreQuery"
              data-selectable
              type="search"
              placeholder="Filter tags…"
              aria-label="Filter tags"
              class="min-w-0 flex-1 bg-transparent text-micro text-fg outline-none placeholder:text-fg-dim"
            />
          </label>
          <div class="max-h-64 overflow-y-auto" role="group" aria-label="All tags">
            <button
              v-for="tag in moreMatches"
              :key="tag.name"
              type="button"
              role="checkbox"
              :aria-checked="isActive(tag.name)"
              class="flex h-7 w-full items-center gap-2 rounded-control px-1 text-left text-sm text-fg hover:bg-accent-tint"
              data-test="more-tag-row"
              :data-tag="tag.name"
              @click="emit('toggleTag', tag.name)"
            >
              <span
                class="border-border-control flex h-4 w-4 shrink-0 items-center justify-center rounded-inner border font-mono text-micro leading-none"
                :class="
                  isActive(tag.name)
                    ? 'border-accent bg-accent text-on-accent'
                    : 'bg-bg-deep text-transparent'
                "
                aria-hidden="true"
              >
                ✓
              </span>
              <span class="min-w-0 flex-1 truncate">{{ tag.name }}</span>
              <span class="font-mono text-micro text-fg-dim">{{ tag.count }}</span>
            </button>
            <p v-if="moreMatches.length === 0" class="px-1 py-1 text-micro text-fg-dim">
              No tags match.
            </p>
          </div>
        </div>
      </Popover>
    </template>

    <div class="flex-1" />

    <button
      v-if="anyFilter"
      type="button"
      class="shrink-0 rounded-control px-1.5 text-micro text-fg-dim hover:text-fg"
      data-test="clear-filters"
      @click="emit('clearFilters')"
    >
      Clear filters
    </button>
    <template v-if="hostChips.length > 1">
      <span class="shrink-0 text-micro text-fg-dim" data-test="made-on-label">Made on</span>
      <button
        v-for="host in hostChips"
        :key="host.key"
        type="button"
        class="ms-lib-chip ms-lib-chip--host"
        :class="hostFilter === host.key ? 'ms-lib-chip--on' : ''"
        data-test="host-chip"
        :data-host="host.key"
        :aria-pressed="hostFilter === host.key"
        @click="emit('update:hostFilter', hostFilter === host.key ? 'all' : host.key)"
      >
        <span class="max-w-32 truncate">{{ host.label }}</span>
        <span class="ms-lib-chip__n">{{ host.count }}</span>
      </button>
    </template>
  </div>
</template>

<style scoped>
/* 22px pill chips — denser than the shared Chip's 6/13 padding so the row
   stays 30px. */
.ms-lib-chip,
:deep(.ms-lib-chip.ms-chip) {
  height: var(--mold-ctl-sm);
  padding: 0 10px;
  gap: 6px;
  font-size: var(--mold-fs-micro);
  display: inline-flex;
  align-items: center;
  white-space: nowrap;
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg);
  color: var(--mold-text-2);
  flex: 0 0 auto;
}

.ms-lib-chip--on,
:deep(.ms-lib-chip.ms-chip[data-on="true"]) {
  border-color: var(--mold-blue);
  color: var(--mold-text);
  background: var(--mold-accent-tint);
  box-shadow: inset 0 0 0 1px var(--mold-blue);
}

.ms-lib-chip--more {
  border-style: dashed;
  color: var(--mold-text-dim);
  cursor: pointer;
}

.ms-lib-chip--more:focus-visible,
.ms-lib-chip--host:focus-visible,
.ms-lib-chip--on:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}

.ms-lib-chip__n {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  opacity: 0.7;
}

:deep(.ms-chip[data-on="true"]) .ms-lib-chip__n {
  color: var(--mold-blue);
}

.ms-lib-vr {
  width: 1px;
  height: 16px;
  background: var(--mold-border);
  margin: 0 2px;
  flex: 0 0 auto;
}

/* Machine chips read as facts, so their name and count are both mono. */
.ms-lib-chip--host {
  font-family: var(--mold-font-mono);
  cursor: pointer;
}
</style>
