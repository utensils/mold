<script setup lang="ts">
/*
 * CollectionsShelf — the Albums scope's horizontally scrolling strip of
 * 150px cover cards plus the dashed "New album" card, ABOVE the grid, which
 * stays mounted beneath it. The new card turns into an inline name input;
 * Enter creates (emit), Escape / empty cancels. Pure: the parent resolves
 * covers and runs the store calls.
 */
import { nextTick, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import Keycap from "@ui/components/Keycap.vue";
import CollectionCard, { type CoverTile } from "./CollectionCard.vue";
import { shiftShortcutLabel } from "../../lib/platform";

// Windows and Linux spell this chord Ctrl+Shift+N, not the mac glyph pair.
const newCollectionChord = shiftShortcutLabel("N");

export interface ShelfCard {
  slug: string;
  name: string;
  count: number;
  hostLabels: string[];
  updatedAt: number | null;
  covers: CoverTile[];
  hidden: boolean;
}

withDefaults(
  defineProps<{
    cards: readonly ShelfCard[];
    /** Hide the New card when no host can organize. */
    canCreate?: boolean;
    busy?: boolean;
    nowMs?: number;
    /** Shown in place of cards when there are none to show. */
    note?: string | null;
  }>(),
  { canCreate: true, busy: false, note: null },
);

const emit = defineEmits<{
  open: [slug: string];
  create: [name: string];
  contextmenu: [slug: string, event: MouseEvent];
}>();

const creating = ref(false);
const draft = ref("");
const inputEl = ref<HTMLInputElement | null>(null);

async function startCreate() {
  creating.value = true;
  draft.value = "";
  await nextTick();
  inputEl.value?.focus();
}

function cancelCreate() {
  creating.value = false;
  draft.value = "";
}

function commitCreate() {
  const name = draft.value.trim();
  if (!name) {
    cancelCreate();
    return;
  }
  emit("create", name);
  creating.value = false;
  draft.value = "";
}

defineExpose({ startCreate, isCreating: () => creating.value });
</script>

<template>
  <div
    class="border-border flex shrink-0 gap-2.5 overflow-x-auto border-b px-3.5 py-3"
    data-test="collections-shelf"
  >
    <p v-if="note" class="self-center text-xs text-fg-dim" data-test="collections-shelf-note">
      {{ note }}
    </p>
    <CollectionCard
      v-for="card in cards"
      :key="card.slug"
      :name="card.name"
      :count="card.count"
      :host-labels="card.hostLabels"
      :updated-at="card.updatedAt"
      :covers="card.covers"
      :hidden="card.hidden"
      :now-ms="nowMs"
      :data-slug="card.slug"
      @open="emit('open', card.slug)"
      @contextmenu="emit('contextmenu', card.slug, $event)"
    />
    <div
      v-if="canCreate"
      class="ms-shelf-card flex flex-col justify-center gap-1.5 rounded-control border border-dashed border-border-control bg-transparent p-2.5 text-left"
      data-test="new-collection-card"
    >
      <template v-if="creating">
        <input
          ref="inputEl"
          v-model="draft"
          data-selectable
          type="text"
          placeholder="Album name"
          aria-label="New album name"
          class="border-border h-7 w-full rounded-control border bg-bg-deep px-2 font-sans font-semibold text-base font-semibold text-fg outline-none focus:border-accent"
          data-test="new-collection-input"
          @keydown.enter.prevent="commitCreate"
          @keydown.esc.prevent.stop="cancelCreate"
          @blur="commitCreate"
        />
        <span class="text-xs text-fg-dim">Enter to create · Esc to cancel</span>
      </template>
      <template v-else>
        <button
          type="button"
          class="flex items-center gap-1.5 text-left text-xs text-fg-dim hover:text-accent"
          data-test="new-collection-label"
          aria-label="New album"
          :disabled="busy"
          @click="startCreate"
        >
          <Icon name="plus" :size="13" />
          New album
        </button>
        <span class="font-mono text-micro text-fg-dim"
          ><Keycap>{{ newCollectionChord }}</Keycap></span
        >
      </template>
    </div>
  </div>
</template>

<style scoped>
/* The mock's strip: fixed 150px cards that scroll sideways, never a grid that
   pushes the pictures off the screen. */
.ms-shelf-card,
:deep(.ms-ccard) {
  width: 150px;
  flex: 0 0 150px;
}
</style>
