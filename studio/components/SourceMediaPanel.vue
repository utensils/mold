<script setup lang="ts">
import { ref, watch } from "vue";
import SourceMediaCard from "./SourceMediaCard.vue";
import {
  retainedSourceMediaInventory,
  retainedSourceMediaDisclosure,
  type RetainedSourceMediaInventory,
} from "../api/gallerySourceMedia";
import {
  sourceOwnerIdentity,
  type SourceMediaOwner,
  type SourceMediaSelection,
} from "../lib/sourceMediaBrowser";
const props = defineProps<{ owner: SourceMediaOwner }>();
const emit = defineEmits<{ use: [selection: SourceMediaSelection] }>();
const inventory = ref<RetainedSourceMediaInventory>();
const error = ref("");
const loading = ref(false);
const retry = ref(0);
watch(
  [() => sourceOwnerIdentity(props.owner), retry],
  async (_, __, cleanup) => {
    const controller = new AbortController();
    cleanup(() => controller.abort());
    inventory.value = undefined;
    error.value = "";
    loading.value = true;
    try {
      const result = await retainedSourceMediaInventory(
        props.owner.filename,
        props.owner.target,
        controller.signal,
      );
      if (!controller.signal.aborted) inventory.value = result;
    } catch (failure) {
      if (!controller.signal.aborted)
        error.value =
          failure instanceof Error ? failure.message : String(failure);
    } finally {
      if (!controller.signal.aborted) loading.value = false;
    }
  },
  { immediate: true },
);
</script>
<template>
  <section
    class="source-panel"
    aria-label="Source media"
    data-test="source-media-panel"
  >
    <h3>Source media</h3>
    <p v-if="loading" role="status">Looking for retained sources…</p>
    <p v-else-if="error" role="status">
      {{ error }} <button type="button" @click="retry++">Retry</button>
    </p>
    <div
      v-else-if="
        inventory?.availability === 'available' && inventory.members?.length
      "
      class="source-panel__grid"
    >
      <SourceMediaCard
        v-for="member in inventory.members"
        :key="member.member_id"
        :selection="{ owner, member }"
        @use="emit('use', $event)"
      />
    </div>
    <p
      v-else-if="
        inventory?.availability === 'unavailable_legacy' ||
        (inventory?.availability === 'available' && !inventory.members?.length)
      "
    >
      No retained source media is available for this print.
    </p>
    <p v-else role="status">
      {{
        (inventory && retainedSourceMediaDisclosure(inventory.availability)) ||
        "Source media is unavailable."
      }}
      <button type="button" @click="retry++">Retry</button>
    </p>
  </section>
</template>
<style scoped>
.source-panel {
  margin-block: 16px;
  min-width: 0;
}
.source-panel h3 {
  font-size: inherit;
  font-weight: 600;
  margin: 0 0 10px;
}
.source-panel p {
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-dim, #888);
}
.source-panel__grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 125px), 1fr));
  gap: 8px;
}
.source-panel button {
  min-height: 44px;
  color: inherit;
}
</style>
