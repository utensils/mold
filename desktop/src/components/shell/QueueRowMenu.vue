<script setup lang="ts">
import VideoJobControl from "@studio/components/VideoJobControl.vue";
/* The ⋯ at the end of a queue row: opens the row's actions as the app's own
 * context menu, anchored to the button. */
import VideoRestartControl from "@studio/components/VideoRestartControl.vue";
import { useHostsStore } from "../../stores/hosts";
import Icon from "@ui/components/Icon.vue";
import { useQueueCommands } from "../../composables/useQueueCommands";
import type { QueueRow } from "../../composables/useQueueActivity";

defineProps<{ row: QueueRow }>();
const commands = useQueueCommands();
const hosts = useHostsStore();
</script>

<template>
  <VideoJobControl
    v-if="row.kind === 'shared' && row.shared.kind === 'video_job'"
    :host-id="row.shared.hostId"
    :job-id="row.shared.id"
    :instance-id="row.shared.instanceId"
  />
  <VideoRestartControl
    :host-id="
      row.kind === 'shared' ? row.shared.hostId : (row.print.hostId ?? hosts.primaryHost?.id)
    "
    :job-id="row.kind === 'shared' ? row.shared.id : (row.print.recoveredJobId ?? row.print.id)"
    :state="row.kind === 'shared' ? row.shared.phase : row.print.status"
    :chain="
      row.kind === 'shared'
        ? row.shared.kind === 'sequence' || row.shared.execution === 'chain'
        : row.print.chainStageCount != null
    "
  />
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
