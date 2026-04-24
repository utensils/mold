<script setup lang="ts">
import { computed } from "vue";
import type { Job } from "../composables/useGenerateStream";
import RunningJobCard from "./RunningJobCard.vue";

// Hide-mode propagates down to each card's thumbnail. Revealed uses job
// ids as keys (not filenames) because running jobs don't have an on-disk
// filename until they complete.
const props = withDefaults(
  defineProps<{
    jobs: Job[];
    hideMode?: boolean;
    revealed?: Set<string>;
  }>(),
  {
    hideMode: false,
    revealed: () => new Set<string>(),
  },
);
const emit = defineEmits<{
  (e: "cancel", id: string): void;
  (e: "open", job: Job): void;
  (e: "dismiss", id: string): void;
  (e: "clear-finished"): void;
  (e: "reveal", id: string): void;
}>();

const hasFinished = computed(() =>
  props.jobs.some((j) => j.state !== "running"),
);
</script>

<template>
  <div v-if="jobs.length" class="mt-4 flex flex-col gap-2">
    <div class="flex gap-3 overflow-x-auto pb-2">
      <RunningJobCard
        v-for="job in jobs"
        :key="job.id"
        :job="job"
        :hide-mode="hideMode"
        :revealed="revealed.has(job.id)"
        @cancel="(id: string) => emit('cancel', id)"
        @open="(j: Job) => emit('open', j)"
        @dismiss="(id: string) => emit('dismiss', id)"
        @reveal="(id: string) => emit('reveal', id)"
      />
    </div>
    <div v-if="hasFinished" class="flex justify-end">
      <button
        type="button"
        class="text-xs text-slate-400 hover:text-slate-200"
        data-test="clear-finished"
        @click="emit('clear-finished')"
      >
        Clear finished
      </button>
    </div>
  </div>
</template>
