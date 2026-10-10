<script setup lang="ts">
import { useHeldQueueTransfer } from "../composables/useHeldQueueTransfer";
defineProps<{
  hostId: string | null | undefined;
  jobId: string | null | undefined;
  state: string;
}>();
const transfer = useHeldQueueTransfer();
</script>
<template>
  <button
    v-if="hostId && jobId && transfer?.canSendState(hostId, state)"
    type="button"
    data-test="queue-move-to"
    :disabled="transfer.busy.value"
    @click.stop="transfer.open(hostId, jobId)"
  >
    Move to…
  </button>
</template>
<style scoped>
button {
  min-height: 44px;
  min-width: 44px;
  padding: 0 12px;
  border: 1px solid var(--line, var(--mold-border-control));
  border-radius: var(--mold-radius-2);
  background: transparent;
  color: inherit;
  font: inherit;
  cursor: pointer;
}
button:disabled {
  opacity: 0.5;
  cursor: default;
}
</style>
