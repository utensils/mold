<script setup lang="ts">
/*
 * BulkBar — the floating selection toolbar at the foot of the My images
 * grid. Everything: N / M selected · Select all · Clear · Add to album ▾
 * (CollectionPicker popover, mixed state over the selection) · Tag ▾
 * (TagEditor popover; chips = the tags every selected picture carries;
 * add / remove apply to the whole selection) · ♥ Favourite (toggle) · Trash
 * (6 s undo) — or, on machines without a trash, the old two-press
 * "Delete N pictures? This can't be undone." arming button. Inside an album
 * drill-in: Remove from album. Trash scope: Restore / Delete forever. Pure
 * props/emits; the view runs the store calls.
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
    <span class="font-mono px-1 text-micro text-fg">
      {{ selectedCount }}
      <span class="text-fg-dim">/ {{ total }} selected</span>
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
            <p class="lightbox-kicker mb-1 px-0.5">Add {{ selectedCount }} {{ noun }} to</p>
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
              :disabled="none || busy || organizeBlocked"
              :title="organizeBlockedReason ?? undefined"
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
            <p class="mt-1 px-0.5 font-mono text-micro text-fg-dim">
              Adding or removing a tag applies to every selected picture.
            </p>
          </div>
        </Popover>
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
          <span aria-hidden="true">♥</span>
          {{ allFavorite && !none ? "Unfavourite" : "Favourite" }}
        </button>
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
      <span class="ms-bb-vr" aria-hidden="true" />
      <button
        type="button"
        class="ms-bb"
        :class="confirming && !trash ? 'ms-bb--armed' : 'hover:text-error'"
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
  border: 1px solid var(--mold-border);
  background: transparent;
  color: var(--mold-text-2);
  border-radius: var(--mold-radius-2);
  padding: 0 10px;
  font-family: var(--mold-font-sans);
  font-size: 11.5px;
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

.ms-bb-vr {
  width: 1px;
  height: 18px;
  background: var(--mold-border);
  margin: 0 2px;
}

.lightbox-kicker {
  font-family: var(--mold-font-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--mold-text-dim);
}
</style>
