<script setup lang="ts">
/*
 * CollectionsShelf — the Albums scope's grid of cover cards plus the
 * dashed "New album" card. The new card turns into an
 * inline name input; Enter creates (emit), Escape / empty cancels. Pure:
 * the parent resolves covers and runs the store calls.
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
  }>(),
  { canCreate: true, busy: false },
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
  <div class="ms-shelf grid gap-4 p-6" data-test="collections-shelf">
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
      class="flex flex-col gap-0.5 rounded-control border border-dashed border-border-control bg-transparent p-2.5 text-left"
      data-test="new-collection-card"
    >
      <button
        type="button"
        class="mb-2 flex aspect-[4/3] w-full items-center justify-center rounded-control border border-dashed border-border-control text-fg-dim transition-colors duration-100 hover:border-accent hover:text-accent focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-50"
        aria-label="New album"
        :disabled="busy"
        @click="startCreate"
      >
        <Icon name="plus" :size="22" />
      </button>
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
          class="font-sans font-semibold text-left text-base font-semibold text-fg hover:text-accent"
          data-test="new-collection-label"
          :disabled="busy"
          @click="startCreate"
        >
          New album
        </button>
        <span class="font-mono text-micro text-fg-dim"
          ><Keycap>{{ newCollectionChord }}</Keycap></span
        >
        <span class="text-xs text-fg-dim">
          Name it, then add pictures from the grid or a selection.
        </span>
      </template>
    </div>
  </div>
</template>

<style scoped>
/* 4-up at the desktop workspace width, 3-up below ~1100px, 2-up when narrow. */
.ms-shelf {
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
}
</style>
