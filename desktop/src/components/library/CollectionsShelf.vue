<script setup lang="ts">
/*
 * CollectionsShelf — the Collections scope's grid of cover cards plus the
 * dashed "New collection" card (V3 "Shelf"). The new card turns into an
 * inline name input; Enter creates (emit), Escape / empty cancels. Pure:
 * the parent resolves covers and runs the store calls.
 */
import { nextTick, ref } from "vue";
import Icon from "@ui/components/Icon.vue";
import Keycap from "@ui/components/Keycap.vue";
import CollectionCard, { type CoverTile } from "./CollectionCard.vue";

export interface ShelfCard {
  slug: string;
  name: string;
  count: number;
  hostLabels: string[];
  updatedAt: number | null;
  covers: CoverTile[];
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
      :now-ms="nowMs"
      :data-slug="card.slug"
      @open="emit('open', card.slug)"
      @contextmenu="emit('contextmenu', card.slug, $event)"
    />
    <div
      v-if="canCreate"
      class="flex flex-col gap-0.5 rounded-card border border-dashed border-ce bg-transparent p-2.5 text-left"
      data-test="new-collection-card"
    >
      <button
        type="button"
        class="mb-2 flex aspect-[4/3] w-full items-center justify-center rounded-[8px] border border-dashed border-ce text-ink-3 transition-colors duration-100 hover:border-safelight hover:text-safelight focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-safelight disabled:opacity-50"
        aria-label="New collection"
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
          placeholder="Collection name"
          aria-label="New collection name"
          class="border-edge h-7 w-full rounded-control border bg-bath px-2 font-display text-[15px] font-semibold text-ink outline-none focus:border-safelight"
          data-test="new-collection-input"
          @keydown.enter.prevent="commitCreate"
          @keydown.esc.prevent.stop="cancelCreate"
          @blur="commitCreate"
        />
        <span class="text-[11.5px] text-ink-3">Enter to create · Esc to cancel</span>
      </template>
      <template v-else>
        <button
          type="button"
          class="font-display text-left text-[15px] font-semibold text-ink hover:text-safelight"
          data-test="new-collection-label"
          :disabled="busy"
          @click="startCreate"
        >
          New collection
        </button>
        <span class="font-utility text-[10.5px] text-ink-3"><Keycap>⌘⇧N</Keycap></span>
        <span class="text-[11.5px] text-ink-3">
          Name it, then add prints from the grid or a selection.
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
