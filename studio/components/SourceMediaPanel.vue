<script setup lang="ts">
import { computed, ref, watch, onBeforeUnmount } from "vue";
import {
  observeSourceMediaEvents,
  sourceEventMatches,
  cachedSourceInventory,
  cacheSourceInventory,
  invalidateSourceOwner,
} from "../lib/sourceMediaResources";
import SourceMediaCard from "./SourceMediaCard.vue";
import {
  retainedSourceMediaInventory,
  retainedSourceMediaDisclosure,
  type RetainedSourceMediaInventory,
} from "../api/gallerySourceMedia";
import {
  sourceOwnerIdentity,
  groupSourceMedia,
  type SourceMediaOwner,
  type SourceMediaSelection,
} from "../lib/sourceMediaBrowser";
const props = defineProps<{ owner: SourceMediaOwner }>();
const emit = defineEmits<{ use: [selection: SourceMediaSelection] }>();
const inventory = ref<RetainedSourceMediaInventory>();
const error = ref("");
const loading = ref(false);
const retry = ref(0);
const page = ref(48);
const groups = computed(() =>
  groupSourceMedia(
    (inventory.value?.members ?? []).map((member) => ({
      owner: props.owner,
      member,
    })),
  ),
);
watch(
  () => groups.value.length,
  (count) => {
    page.value = Math.min(page.value, Math.max(48, Math.ceil(count / 48) * 48));
  },
);
watch(
  [() => sourceOwnerIdentity(props.owner), retry],
  async (_, __, cleanup) => {
    const controller = new AbortController();
    cleanup(() => controller.abort());
    inventory.value = undefined;
    error.value = "";
    loading.value = true;
    try {
      const key = sourceOwnerIdentity(props.owner);
      if (retry.value) invalidateSourceOwner(key, true);
      const result =
        cachedSourceInventory(key) ??
        (await retainedSourceMediaInventory(
          props.owner.filename,
          props.owner.target,
          controller.signal,
        ));
      if (!controller.signal.aborted) {
        inventory.value = result;
        cacheSourceInventory(key, result);
      }
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
const stopEvents = observeSourceMediaEvents((event) => {
  if (sourceEventMatches(props.owner, event)) retry.value++;
});
const refreshTimer = setInterval(() => {
  retry.value++;
}, 60_000);
onBeforeUnmount(() => {
  clearInterval(refreshTimer);
  stopEvents();
});
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
        v-for="group in groups.slice(page - 48, page)"
        :key="group.key"
        :selection="group.usages[0]!"
        :usages="group.usages"
        @use="emit('use', $event)"
      />
      <button v-if="page > 48" type="button" @click="page -= 48">
        Previous sources
      </button>
      <button v-if="groups.length > page" type="button" @click="page += 48">
        Next sources
      </button>
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
