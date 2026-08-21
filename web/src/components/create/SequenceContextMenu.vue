<script setup lang="ts">
/*
 * Web's sequence right-click menu. Desktop has an app-wide custom context
 * menu store; web renders each menu inline (the pattern MachinesPage and
 * CollectionsShelf already use), so this is that pattern made reusable for
 * the sequence bench. The ENTRIES themselves come from the shared
 * `studio/lib/sequenceContextMenu` builder, so the two surfaces offer the
 * same actions in the same order.
 */
import { computed, onBeforeUnmount, onMounted } from "vue";
import {
  isSequenceMenuSeparator,
  type SequenceMenuEntry,
} from "@studio/lib/sequenceContextMenu";

const props = defineProps<{
  entries: SequenceMenuEntry[];
  /** Client coordinates of the originating right-click. */
  x: number;
  y: number;
}>();

const emit = defineEmits<{ close: [] }>();

const MENU_WIDTH = 216;
const ITEM_HEIGHT = 32;
const SEPARATOR_HEIGHT = 9;
const PANEL_PADDING = 10;
const MARGIN = 6;

const estimatedHeight = computed(
  () =>
    props.entries.reduce(
      (height, entry) =>
        height +
        (isSequenceMenuSeparator(entry) ? SEPARATOR_HEIGHT : ITEM_HEIGHT),
      0,
    ) + PANEL_PADDING,
);

/** Keep the panel on screen — a right-click near the edge must not push it
 *  out of the viewport with no way back. */
const position = computed(() => {
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  const width = Math.min(MENU_WIDTH, Math.max(0, viewportWidth - MARGIN * 2));
  const height = Math.min(
    estimatedHeight.value,
    Math.max(0, viewportHeight - MARGIN * 2),
  );
  return {
    left: `${Math.max(MARGIN, Math.min(props.x, viewportWidth - width - MARGIN))}px`,
    top: `${Math.max(MARGIN, Math.min(props.y, viewportHeight - height - MARGIN))}px`,
  };
});

function run(entry: SequenceMenuEntry) {
  if (isSequenceMenuSeparator(entry) || entry.disabled) return;
  // Close first: an action may open a confirm dialog or a file picker.
  emit("close");
  entry.action();
}

function onDocumentPointer(event: Event) {
  const node = event.target as HTMLElement | null;
  if (node?.closest?.("[data-test='sequence-context-menu']")) return;
  emit("close");
}

function onWindowKey(event: KeyboardEvent) {
  if (event.key === "Escape") emit("close");
}

onMounted(() => {
  document.addEventListener("pointerdown", onDocumentPointer);
  window.addEventListener("keydown", onWindowKey);
});

onBeforeUnmount(() => {
  document.removeEventListener("pointerdown", onDocumentPointer);
  window.removeEventListener("keydown", onWindowKey);
});
</script>

<template>
  <div
    class="sq-context"
    data-test="sequence-context-menu"
    role="menu"
    :style="position"
  >
    <template v-for="(entry, index) in entries" :key="index">
      <div
        v-if="isSequenceMenuSeparator(entry)"
        class="sq-context__separator"
      />
      <button
        v-else
        type="button"
        role="menuitem"
        data-test="sequence-context-item"
        :class="{ 'sq-context__danger': entry.danger }"
        :disabled="entry.disabled"
        @click="run(entry)"
      >
        {{ entry.label }}
      </button>
    </template>
  </div>
</template>

<style scoped>
.sq-context {
  position: fixed;
  z-index: 80;
  min-width: 210px;
  padding: 5px;
  border: 1px solid var(--ce);
  border-radius: 9px;
  background: var(--bench);
  box-shadow: 0 14px 36px color-mix(in srgb, var(--bath) 50%, transparent);
}

.sq-context button {
  display: block;
  width: 100%;
  min-height: 32px;
  padding: 0 10px;
  border: 0;
  border-radius: 5px;
  background: transparent;
  color: var(--rebate);
  text-align: left;
  font-size: 12.5px;
  cursor: pointer;
}

.sq-context button:hover:not(:disabled) {
  background: color-mix(in srgb, var(--safelight) 14%, transparent);
}

.sq-context button:disabled {
  color: var(--ink-3);
  cursor: default;
}

.sq-context__separator {
  height: 1px;
  margin: 4px 6px;
  background: var(--ce);
}

.sq-context .sq-context__danger {
  color: var(--stop);
}
</style>
