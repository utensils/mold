<script setup lang="ts">
/*
 * BulkBar — the floating selection toolbar at the foot of the Library grid
 * (V3 "Shelf"). Prints: N / M selected · Select all · Clear · Add to
 * collection ▾ (CollectionPicker popover, mixed state over the selection) ·
 * Tag ▾ (TagEditor popover; chips = the tags every selected print carries;
 * add / remove apply to the whole selection) · ♥ Favorite (toggle) · Trash
 * (6 s undo) — or, on hosts without a trash, the old two-press
 * "Delete N prints? This can't be undone." arming button. Inside a
 * collection drill-in: Remove from collection. Trash scope: Restore /
 * Delete forever. Pure props/emits; the view runs the store calls.
 */
import { computed, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import Keycap from "@ui/components/Keycap.vue";
import Popover from "@ui/components/Popover.vue";
import CollectionPicker from "./CollectionPicker.vue";
import TagEditor from "./TagEditor.vue";
import type { LibraryScope } from "../../stores/gallery";
import type { MergedCollection } from "@studio/lib/libraryOrganization";
import type { TagCount } from "@studio/lib/api/galleryOrganization";

const props = withDefaults(
  defineProps<{
    selectedCount: number;
    total: number;
    scope: LibraryScope;
    /** Organization affordances (collections / tags / ♥) are available. */
    organize?: boolean;
    /** Every selected print lives on trash-capable hosts ⇒ "Move to trash". */
    trash?: boolean;
    /** Two-press arming state for the hard-delete path (parent-owned so the
     *  context menu and keyboard can arm it too). */
    confirming?: boolean;
    busy?: boolean;
    collections?: readonly MergedCollection[];
    /** Slugs every selected print is in. */
    collectionSelected?: readonly string[];
    /** Slugs only some selected prints are in. */
    collectionMixed?: readonly string[];
    collectionCounts?: ((slug: string) => number) | null;
    /** Tags every selected print carries (intersection). */
    tags?: readonly string[];
    tagSuggestions?: readonly TagCount[];
    /** Every selected print is already a favorite ⇒ the toggle unfavorites. */
    allFavorite?: boolean;
    /** Open collection (drill-in) ⇒ offer Remove from collection. */
    collectionName?: string | null;
    /** Host labels the fan-out reaches ("This Mac · plato"). */
    hostNote?: string | null;
  }>(),
  {
    organize: false,
    trash: false,
    confirming: false,
    busy: false,
    collections: () => [],
    collectionSelected: () => [],
    collectionMixed: () => [],
    collectionCounts: null,
    tags: () => [],
    tagSuggestions: () => [],
    allFavorite: false,
    collectionName: null,
    hostNote: null,
  },
);

const emit = defineEmits<{
  selectAll: [];
  clear: [];
  exit: [];
  favorite: [value: boolean];
  trash: [];
  /** Hard delete (non-trash hosts): first press arms, second deletes. */
  "update:confirming": [value: boolean];
  delete: [];
  restore: [];
  deleteForever: [];
  removeFromCollection: [];
  toggleCollection: [slug: string, checked: boolean];
  createCollection: [name: string];
  addTags: [names: string[]];
  removeTags: [names: string[]];
}>();

const none = computed(() => props.selectedCount === 0);
const noun = computed(() => (props.selectedCount === 1 ? "print" : "prints"));

const collectionsOpen = ref(false);
const tagsOpen = ref(false);

function openCollections() {
  tagsOpen.value = false;
  collectionsOpen.value = true;
}
function openTags() {
  collectionsOpen.value = false;
  tagsOpen.value = true;
}
/** Close whichever popover is open; true when one was. */
function closePopovers(): boolean {
  const was = collectionsOpen.value || tagsOpen.value;
  collectionsOpen.value = false;
  tagsOpen.value = false;
  return was;
}

function onDeleteClick() {
  if (props.trash) {
    emit("trash");
    return;
  }
  if (!props.confirming) {
    emit("update:confirming", true);
    return;
  }
  // The parent disarms as part of running the delete.
  emit("delete");
}

defineExpose({ openCollections, openTags, closePopovers });
</script>

<template>
  <div
    data-test="bulk-action-bar"
    class="border-edge absolute bottom-4 left-1/2 z-30 flex max-w-[calc(100%-2rem)] -translate-x-1/2 flex-wrap items-center gap-2 rounded-chrome border bg-bench px-3 py-2 shadow-lg"
    role="toolbar"
    aria-label="Selection actions"
  >
    <span class="data-mono px-1 text-caption text-ink">
      {{ selectedCount }}
      <span class="text-ink-3">/ {{ total }} selected</span>
    </span>
    <button
      type="button"
      class="ms-bb"
      :disabled="total === 0"
      data-test="bulk-select-all"
      @click="emit('selectAll')"
    >
      Select all
    </button>
    <button
      type="button"
      class="ms-bb"
      :disabled="none"
      data-test="bulk-clear"
      @click="emit('clear')"
    >
      Clear
    </button>

    <template v-if="scope === 'trash'">
      <span class="ms-bb-vr" aria-hidden="true" />
      <button
        type="button"
        class="ms-bb"
        :disabled="none || busy"
        data-test="bulk-restore"
        @click="emit('restore')"
      >
        <Icon name="reuse" :size="13" />
        Restore
      </button>
      <button
        type="button"
        class="ms-bb ms-bb--danger"
        :disabled="none || busy"
        data-test="bulk-delete-forever"
        @click="emit('deleteForever')"
      >
        Delete forever
      </button>
    </template>

    <template v-else>
      <template v-if="organize">
        <span class="ms-bb-vr" aria-hidden="true" />
        <Popover
          :open="collectionsOpen"
          placement="top-start"
          label="Add to collection"
          @update:open="collectionsOpen = $event"
        >
          <template #trigger>
            <button
              type="button"
              class="ms-bb"
              :class="{ 'ms-bb--on': collectionsOpen }"
              :disabled="none || busy"
              :aria-expanded="collectionsOpen"
              data-test="bulk-collections"
              @click="collectionsOpen ? closePopovers() : openCollections()"
            >
              <Icon name="collection" :size="13" />
              Add to collection
              <Icon name="chevron-down" :size="11" />
            </button>
          </template>
          <div class="w-60" data-test="bulk-collections-panel">
            <p class="lightbox-kicker mb-1 px-0.5">Add {{ selectedCount }} {{ noun }} to</p>
            <CollectionPicker
              :collections="[...collections]"
              :selected="[...collectionSelected]"
              :mixed="[...collectionMixed]"
              :counts="collectionCounts"
              :disabled="busy"
              :host-note="hostNote ? `fans out to ${hostNote}` : null"
              aria-label="Add to collection"
              @toggle="(slug, checked) => emit('toggleCollection', slug, checked)"
              @create="(name) => emit('createCollection', name)"
            />
          </div>
        </Popover>
        <Popover
          :open="tagsOpen"
          placement="top-start"
          label="Tag selection"
          @update:open="tagsOpen = $event"
        >
          <template #trigger>
            <button
              type="button"
              class="ms-bb"
              :class="{ 'ms-bb--on': tagsOpen }"
              :disabled="none || busy"
              :aria-expanded="tagsOpen"
              data-test="bulk-tags"
              @click="tagsOpen ? closePopovers() : openTags()"
            >
              <Icon name="tag" :size="13" />
              Tag
              <Icon name="chevron-down" :size="11" />
            </button>
          </template>
          <div class="w-64" data-test="bulk-tags-panel">
            <p class="lightbox-kicker mb-1 px-0.5">Tags on all {{ selectedCount }} {{ noun }}</p>
            <TagEditor
              :model-value="[...tags]"
              :suggestions="[...tagSuggestions]"
              :disabled="busy"
              aria-label="Tag the selection"
              @add="(name) => emit('addTags', [name])"
              @remove="(name) => emit('removeTags', [name])"
            />
            <p class="mt-1 px-0.5 font-utility text-[9.5px] text-ink-3">
              Adding or removing a tag applies to every selected print.
            </p>
          </div>
        </Popover>
        <button
          type="button"
          class="ms-bb"
          :class="{ 'ms-bb--on': allFavorite && !none }"
          :disabled="none || busy"
          :aria-pressed="allFavorite && !none"
          data-test="bulk-favorite"
          @click="emit('favorite', !allFavorite)"
        >
          <span aria-hidden="true">♥</span>
          {{ allFavorite && !none ? "Unfavorite" : "Favorite" }}
        </button>
        <button
          v-if="collectionName"
          type="button"
          class="ms-bb"
          :disabled="none || busy"
          data-test="bulk-remove-from-collection"
          @click="emit('removeFromCollection')"
        >
          <Icon name="close" :size="12" />
          Remove from collection
        </button>
      </template>
      <span class="ms-bb-vr" aria-hidden="true" />
      <button
        type="button"
        class="ms-bb"
        :class="confirming && !trash ? 'ms-bb--armed' : 'hover:text-stop'"
        :disabled="none || busy"
        data-test="bulk-delete"
        @blur="!trash && emit('update:confirming', false)"
        @click="onDeleteClick"
      >
        <Icon v-if="!confirming || trash" name="trash" :size="13" />
        {{
          busy
            ? trash
              ? "Moving…"
              : "Deleting…"
            : trash
              ? `Move ${selectedCount} ${noun} to trash`
              : confirming
                ? `Delete ${selectedCount} ${noun}? This can't be undone.`
                : "Delete selected"
        }}
        <Keycap v-if="trash && !busy">⌫</Keycap>
      </button>
    </template>

    <button
      type="button"
      class="flex h-7 w-7 items-center justify-center rounded-control text-ink-3 transition-colors duration-100 hover:text-ink"
      aria-label="Exit select mode"
      title="Exit select mode (Esc)"
      @click="emit('exit')"
    >
      ✕
    </button>
  </div>
</template>

<style scoped>
.ms-bb {
  height: 28px;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  border: 1px solid var(--edge);
  background: transparent;
  color: var(--ink-2);
  border-radius: var(--radius-control);
  padding: 0 10px;
  font-family: var(--f-body);
  font-size: 11.5px;
  white-space: nowrap;
  transition:
    color var(--dur-quick) var(--ease),
    border-color var(--dur-quick) var(--ease),
    background var(--dur-quick) var(--ease);
}

.ms-bb:hover:not(:disabled) {
  color: var(--rebate);
}

.ms-bb:disabled {
  opacity: 0.5;
}

.ms-bb--on {
  border-color: var(--safelight);
  color: var(--safelight);
  background: var(--sel-bg);
}

.ms-bb--danger {
  color: var(--stop);
  border-color: color-mix(in srgb, var(--stop) 50%, transparent);
}

.ms-bb--armed {
  border-color: var(--stop);
  background: var(--stop);
  color: var(--on-accent);
  font-weight: 600;
}

.ms-bb:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-bb-vr {
  width: 1px;
  height: 18px;
  background: var(--edge);
  margin: 0 2px;
}

.lightbox-kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
}
</style>
