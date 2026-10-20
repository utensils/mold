<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import { useHeldQueueTransfer } from "../composables/useHeldQueueTransfer";
import { createVideoRestart } from "../composables/useVideoRestart";
const props = defineProps<{
  hostId?: string | null | undefined;
  jobId?: string | null | undefined;
  state: string;
  chain: boolean;
}>();
const transfer = useHeldQueueTransfer();
const state = createVideoRestart(
  computed(() => transfer?.hosts.value ?? []),
  transfer?.busy ?? ref(false),
);
const {
  active,
  loading,
  busy,
  cancelling,
  error,
  source,
  options,
  selected,
  destinations,
  admission,
  detail,
  settled,
  filename,
  status,
} = state;
const eligible = computed(
  () =>
    transfer &&
    props.chain &&
    props.hostId &&
    props.jobId &&
    ["failed", "error", "paused", "interrupted"].includes(props.state),
);
const sameMachine = computed(
  () =>
    destinations.value.find((host) => host.id === selected.value)
      ?.instanceId === source.value?.instanceId,
);
let timer: ReturnType<typeof setTimeout> | undefined;
watch([active, admission, settled], () => {
  if (timer) clearTimeout(timer);
  if (!active.value || !admission.value || settled.value) return;
  const poll = async () => {
    await state.refresh();
    if (active.value && admission.value && !settled.value)
      timer = setTimeout(poll, 2500);
  };
  timer = setTimeout(poll, 2500);
});
onBeforeUnmount(() => {
  state.close();
  if (timer) clearTimeout(timer);
});
</script>
<template>
  <button
    v-if="eligible"
    type="button"
    class="video-restart-open"
    data-test="video-restart-open"
    @click.stop="state.open(hostId!, jobId!)"
  >
    Start again…
  </button>
  <ModalPanel
    :open="active"
    label="Start video again"
    :width="480"
    @close="state.close"
  >
    <section class="video-restart" data-test="video-restart">
      <h2>Start video again</h2>
      <p>
        Create a new attempt with the same prompts, seed, settings, and saved
        inputs. The original stays on its machine.
      </p>
      <p v-if="loading" role="status">Checking this video’s saved inputs…</p>
      <template v-else-if="!admission && options">
        <label v-if="destinations.length"
          >Machine
          <select
            v-model="selected"
            :disabled="busy"
            aria-label="Restart destination"
          >
            <option
              v-for="host in destinations"
              :key="host.id"
              :value="host.id"
            >
              {{ host.label
              }}{{
                host.instanceId === source?.instanceId
                  ? " · original machine"
                  : ""
              }}
            </option>
          </select>
        </label>
        <p v-if="!options.restart.supported">{{ options.restart.reason }}</p>
        <p v-if="!options.send.supported">{{ options.send.reason }}</p>
      </template>
      <p v-if="status" role="status">{{ status }}</p>
      <p v-if="detail?.error" role="alert">{{ detail.error }}</p>
      <p v-if="filename">Find the new video in the destination’s Library.</p>
      <p v-if="admission" class="video-restart__operation">
        Job {{ admission.operationId }}
      </p>
      <p v-if="error" role="alert" class="video-restart__error">{{ error }}</p>
      <div class="video-restart__actions">
        <button type="button" @click="state.close">
          {{ admission ? "Done" : "Cancel" }}
        </button>
        <button
          v-if="admission && !settled"
          type="button"
          :disabled="cancelling"
          @click="state.cancel"
        >
          {{ cancelling ? "Stopping…" : "Stop new attempt" }}
        </button>
        <button
          v-if="admission"
          type="button"
          :disabled="loading"
          @click="state.refresh"
        >
          Check queue
        </button>
        <button
          v-else-if="options && destinations.length"
          type="button"
          :disabled="busy || !selected"
          data-test="video-restart-submit"
          @click="state.start"
        >
          {{ busy ? "Starting…" : sameMachine ? "Start again" : "Send video" }}
        </button>
      </div>
    </section>
  </ModalPanel>
</template>
<style scoped>
.video-restart {
  display: grid;
  gap: 16px;
  padding: 20px;
  color: var(--mold-text);
}
.video-restart h2 {
  margin: 0;
  font-size: var(--mold-fs-lg);
}
.video-restart p {
  margin: 0;
  line-height: 1.5;
}
.video-restart label {
  display: grid;
  gap: 8px;
}
.video-restart select,
.video-restart button,
.video-restart-open {
  min-height: 44px;
  padding: 8px 12px;
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
  color: inherit;
  background: var(--mold-bg);
  font: inherit;
}
.video-restart select {
  font-size: max(
    16px,
    var(--mold-fs-md)
  ); /* literal: prevent mobile input zoom */
}
.video-restart__actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.video-restart__operation {
  overflow-wrap: anywhere;
  color: var(--mold-text-dim);
}
.video-restart .video-restart__error {
  color: var(--mold-error);
}
</style>
