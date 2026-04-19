<script setup lang="ts">
import type { Job } from "../composables/useGenerateStream";
import RunningJobCard from "./RunningJobCard.vue";

defineProps<{ jobs: Job[] }>();
const emit = defineEmits<{ (e: "cancel", id: string): void }>();
</script>

<template>
  <div v-if="jobs.length" class="mt-4 flex gap-3 overflow-x-auto pb-2">
    <RunningJobCard
      v-for="job in jobs"
      :key="job.id"
      :job="job"
      @cancel="(id: string) => emit('cancel', id)"
    />
  </div>
</template>
