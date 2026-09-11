<script setup lang="ts">
/*
 * Where the Create rail is drawn. At 900px and above it is the page's second
 * grid column; below that the settings column leaves the page entirely — the
 * web mock's Width rule — and the SAME markup is drawn inside one sheet.
 *
 * It exists so the rail is authored once. Writing it twice is how the narrow
 * board drifted from the wide one in the first place.
 *
 * `ui/` sheets are `position: absolute; inset: 0` by design, so on a scrolling
 * web page they need this fixed viewport host or they render off-screen. The
 * `full` variant silently drops a `#header` slot, so head rows belong in the
 * body, which is exactly what the slot is.
 */
import SheetPanel from "@ui/components/SheetPanel.vue";

defineProps<{
  /** Draw as a sheet over the page rather than as the grid's second column. */
  sheet: boolean;
  /** Sheet only: whether it is showing. */
  open: boolean;
  title?: string;
}>();

const emit = defineEmits<{ close: [] }>();
</script>

<template>
  <div
    v-if="sheet && open"
    class="fixed inset-0 z-40"
    data-test="create-rail-sheet"
  >
    <SheetPanel :open="open" :title="title" @close="emit('close')">
      <div class="rail-body">
        <slot />
      </div>
    </SheetPanel>
  </div>
  <div v-else-if="!sheet" class="rail" data-test="create-rail">
    <slot />
  </div>
</template>

<style scoped>
.rail {
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  gap: 12px;
}

/* `min-height: 0` on every wrapper, and never an inner height: an inner
 * height floors a flex child and takes the sheet's scrollbar with it. */
.rail-body {
  display: flex;
  flex-direction: column;
  min-height: 0;
  gap: 12px;
}
.rail-body > * {
  min-height: 0;
}
</style>
