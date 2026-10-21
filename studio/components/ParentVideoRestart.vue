<script setup lang="ts">
import { computed, inject, onBeforeUnmount } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import { useHeldQueueTransfer } from "../composables/useHeldQueueTransfer";
import { createParentVideoRestart } from "../composables/useParentVideoRestart";
import type { VideoJobOwner } from "../api/videoJobs";
import { VIDEO_LIBRARY_NAVIGATION } from "../lib/videoLibraryNavigation";
const props = defineProps<{ owner: VideoJobOwner; jobId: string }>();
const fleet = useHeldQueueTransfer();
const openLibrary = inject(VIDEO_LIBRARY_NAVIGATION, null);
const state = createParentVideoRestart(
  computed(() => fleet?.hosts.value ?? []),
);
const {
  active,
  loading,
  busy,
  error,
  progress,
  options,
  selected,
  destinations,
  run,
} = state;
const { admission, detail, owner: destination } = run;
onBeforeUnmount(() => {
  state.close();
  run.dispose();
});
</script>
<template>
  <button type="button" @click.stop="state.open(owner, jobId)">
    Start again or send…
  </button>
  <ModalPanel :open="active" label="Start video again" @close="state.close">
    <h2>Start video again</h2>
    <p>
      Make a new attempt with the same prompts, settings, and saved inputs. The
      original stays on its machine.
    </p>
    <p v-if="loading" role="status">Checking saved inputs…</p>
    <template v-else-if="options && !admission && !detail">
      <label
        >Machine
        <select
          v-model="selected"
          :disabled="busy"
          aria-label="Video restart destination"
        >
          <option
            v-for="machine in destinations"
            :key="machine.id"
            :value="machine.id"
          >
            {{ machine.label
            }}{{
              machine.instanceId === owner.instanceId
                ? " · original machine"
                : ""
            }}
          </option>
        </select></label
      >
      <p v-if="!options.restart.supported">{{ options.restart.reason }}</p>
      <p v-if="!options.send.supported">{{ options.send.reason }}</p>
      <button type="button" :disabled="busy || !selected" @click="state.start">
        {{ busy ? "Preparing…" : "Start new attempt" }}
      </button>
    </template>
    <p v-if="progress" role="status">{{ progress }}</p>
    <p v-if="error || run.error.value" role="alert">
      {{ error || run.error.value }}
    </p>
    <template v-if="destination && (admission || detail)">
      <p v-if="detail" role="status">
        {{ detail.state }} · {{ detail.completed_clips }} of
        {{ detail.plan.clip_count }} clips
      </p>
      <button type="button" :disabled="run.busy.value" @click="run.refresh">
        Check queue
      </button>
      <button
        v-if="!detail || detail.can_cancel"
        type="button"
        :disabled="run.cancelling.value"
        @click="run.cancel"
      >
        Stop new attempt
      </button>
      <button
        v-if="detail?.output_filename && openLibrary"
        type="button"
        @click="
          openLibrary({ owner: destination, filename: detail.output_filename })
        "
      >
        View in Library
      </button>
    </template>
    <button type="button" :disabled="busy" @click="state.close">Done</button>
  </ModalPanel>
</template>
