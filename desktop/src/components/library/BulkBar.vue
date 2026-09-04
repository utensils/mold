<script setup lang="ts">
/*
 * BulkBar — the selection toolbar at the foot of the My images grid. "N
 * selected" and the quiet Select all / Clear lead; a spacer then pushes the
 * action cluster right: ★ Favourite (toggle) · Add tag ▾ (TagEditor popover;
 * chips = the tags every selected picture carries; add / remove apply to the
 * whole selection) · Add to album ▾ (CollectionPicker popover, mixed state
 * over the selection) · Export… · Delete — a trash move with its 6 s undo, or
 * on machines without a trash the two-press "Delete N pictures? This can't be
 * undone." arming button. Inside an album drill-in: Remove from album. Trash
 * scope: Restore / Delete forever. Pure props/emits; the view runs the store
 * calls.
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
    /** Organization affordances (albums / tags / ♥) are available. */
    organize?: boolean;
    /** Why the CURRENT selection can't be organized (some selected print has
     *  no copy on an organize-capable host); null when it can. Disables the
     *  organize controls with this reason as their tooltip. */
    organizeBlockedReason?: string | null;
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
    /** Every selected picture is already a favourite ⇒ the toggle unfavourites. */
    allFavorite?: boolean;
    /** Open album (drill-in) ⇒ offer Remove from album. */
    collectionName?: string | null;
    /** Host labels the fan-out reaches ("This Mac · plato"). */
    hostNote?: string | null;
  }>(),
  {
    organize: false,
    organizeBlockedReason: null,
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
  export: [];
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
const noun = computed(() => (props.selectedCount === 1 ? "picture" : "pictures"));
/** Organize controls act on the selection; a blocked selection disables
 *  them with the reason as tooltip (never a silent no-op mutation). */
const organizeBlocked = computed(() => props.organizeBlockedReason != null);

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
    class="border-border flex shrink-0 flex-wrap items-center gap-2 border-t bg-chrome px-3.5 py-2.5"
    role="toolbar"
    aria-label="Selection actions"
  >
    <span class="px-1 text-xs font-semibold text-fg" data-test="bulk-count">
      {{ selectedCount }} selected
    </span>
    <button
      type="button"
      class="ms-bb ms-bb--quiet"
      :disabled="total === 0"
      data-test="bulk-select-all"
      :title="`Select all ${total}`"
      @click="emit('selectAll')"
    >
      Select all
    </button>
    <button
      type="button"
      class="ms-bb ms-bb--quiet"
      :disabled="none"
      data-test="bulk-clear"
      @click="emit('clear')"
    >
      Clear
    </button>

    <span class="flex-1" data-test="bulk-spacer" />

    <template v-if="scope === 'trash'">
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
        <button
          type="button"
          class="ms-bb"
          :class="{ 'ms-bb--on': allFavorite && !none }"
          :disabled="none || busy || organizeBlocked"
          :title="organizeBlockedReason ?? undefined"
          :aria-pressed="allFavorite && !none"
          data-test="bulk-favorite"
          @click="emit('favorite', !allFavorite)"
        >
          <span class="font-mono" aria-hidden="true">★</span>
          {{ allFavorite && !none ? "Unfavourite" : "Favourite" }}
        </button>
        <Popover
          :open="tagsOpen"
          placement="top-start"
          label="Add tag"
          @update:open="tagsOpen = $event"
        >
          <template #trigger>
            <button
              type="button"
              class="ms-bb"
              :class="{ 'ms-bb--on': tagsOpen }"
              :disabled="none || busy || organizeBlocked"
              :title="organizeBlockedReason ?? undefined"
              :aria-expanded="tagsOpen"
              data-test="bulk-tags"
              @click="tagsOpen ? closePopovers() : openTags()"
            >
              <Icon name="tag" :size="13" />
              Add tag
              <Icon name="chevron-down" :size="11" />
            </button>
          </template>
          <div class="w-64" data-test="bulk-tags-panel">
            <p class="ms-group-label mb-1 px-0.5 uppercase">
              Tags on all {{ selectedCount }} {{ noun }}
            </p>
            <TagEditor
              :model-value="[...tags]"
              :suggestions="[...tagSuggestions]"
              :disabled="busy"
              aria-label="Tag the selection"
              @add="(name) => emit('addTags', [name])"
              @remove="(name) => emit('removeTags', [name])"
            />
            <p class="mt-1 px-0.5 font-mono text-micro text-fg-dim">
              Adding or removing a tag applies to every selected picture.
            </p>
          </div>
        </Popover>
        <Popover
          :open="collectionsOpen"
          placement="top-start"
          label="Add to album"
          @update:open="collectionsOpen = $event"
        >
          <template #trigger>
            <button
              type="button"
              class="ms-bb"
              :class="{ 'ms-bb--on': collectionsOpen }"
              :disabled="none || busy || organizeBlocked"
              :title="organizeBlockedReason ?? undefined"
              :aria-expanded="collectionsOpen"
              data-test="bulk-collections"
              @click="collectionsOpen ? closePopovers() : openCollections()"
            >
              <Icon name="collection" :size="13" />
              Add to album
              <Icon name="chevron-down" :size="11" />
            </button>
          </template>
          <div class="w-60" data-test="bulk-collections-panel">
            <p class="ms-group-label mb-1 px-0.5 uppercase">
              Add {{ selectedCount }} {{ noun }} to
            </p>
            <CollectionPicker
              :collections="[...collections]"
              :selected="[...collectionSelected]"
              :mixed="[...collectionMixed]"
              :counts="collectionCounts"
              :disabled="busy"
              :host-note="hostNote ? `fans out to ${hostNote}` : null"
              aria-label="Add to album"
              @toggle="(slug, checked) => emit('toggleCollection', slug, checked)"
              @create="(name) => emit('createCollection', name)"
            />
          </div>
        </Popover>
        <button
          v-if="collectionName"
          type="button"
          class="ms-bb"
          :disabled="none || busy || organizeBlocked"
          :title="organizeBlockedReason ?? undefined"
          data-test="bulk-remove-from-collection"
          @click="emit('removeFromCollection')"
        >
          <Icon name="close" :size="12" />
          Remove from album
        </button>
      </template>
      <button
        type="button"
        class="ms-bb"
        :disabled="none || busy"
        data-test="bulk-export"
        @click="emit('export')"
      >
        Export…
      </button>
      <button
        type="button"
        class="ms-bb ms-bb--delete"
        :class="confirming && !trash ? 'ms-bb--armed' : ''"
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
      class="flex h-7 w-7 items-center justify-center rounded-control text-fg-dim transition-colors duration-100 hover:text-fg"
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
  border: var(--mold-bw) solid var(--mold-border);
  background: transparent;
  color: var(--mold-text-2);
  border-radius: var(--mold-radius-2);
  padding: 0 10px;
  font-family: var(--mold-font-sans);
  font-size: var(--mold-fs-xs);
  white-space: nowrap;
  transition:
    color var(--mold-dur-quick) var(--mold-ease-out),
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    background var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-bb:hover:not(:disabled) {
  color: var(--mold-text);
}

.ms-bb:disabled {
  opacity: 0.5;
}

.ms-bb--on {
  border-color: var(--mold-blue);
  color: var(--mold-blue);
  background: var(--mold-accent-tint);
}

/* The leading Select all / Clear are affordances, not actions: no border, so
   the bordered cluster on the right reads as the one place things happen. */
.ms-bb--quiet {
  border-color: transparent;
  color: var(--mold-text-dim);
  padding: 0 6px;
}

.ms-bb--quiet:hover:not(:disabled) {
  color: var(--mold-text);
}

.ms-bb--delete {
  border-color: var(--mold-error);
  color: var(--mold-error);
  font-weight: 600;
}

.ms-bb--delete:hover:not(:disabled) {
  background: color-mix(in srgb, var(--mold-error) 12%, transparent);
  color: var(--mold-error);
}

.ms-bb--danger {
  color: var(--mold-error);
  border-color: color-mix(in srgb, var(--mold-error) 50%, transparent);
}

.ms-bb--armed {
  border-color: var(--mold-error);
  background: var(--mold-error);
  color: var(--mold-on-accent);
  font-weight: 600;
}

.ms-bb:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}
</style>
