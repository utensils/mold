<script setup lang="ts">
/* The ⋯ at the end of a queue row: opens the row's actions as the app's own
 * context menu, anchored to the button. */
import Icon from "@ui/components/Icon.vue";
import { useQueueCommands } from "../../composables/useQueueCommands";
import type { QueueRow } from "../../composables/useQueueActivity";

defineProps<{ row: QueueRow }>();
const commands = useQueueCommands();
</script>

<template>
  <button
    type="button"
    data-test="queue-row-menu"
    class="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-control text-fg-dim transition-colors duration-100 hover:bg-surface-2 hover:text-fg"
    title="Jump the line, stop, remove…"
    aria-label="Row actions"
    @click.stop="commands.contextMenu($event, row)"
  >
    <Icon name="more" :size="13" />
  </button>
</template>
