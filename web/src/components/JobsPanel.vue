<script setup lang="ts">
import type { ChainJobDetail } from "../types";
import ChainJobCard from "./ChainJobCard.vue";

withDefaults(
  defineProps<{
    jobs?: ChainJobDetail[];
  }>(),
  {
    jobs: () => [],
  },
);
</script>

<template>
  <section data-test="jobs-panel" class="flex flex-col gap-2">
    <!-- TODO(BR55 Phase 6): Load durable chain job listing and synchronize selected job detail with useChainJobStream. -->
    <ChainJobCard v-for="job in jobs" :key="job.id" :job="job" />
  </section>
</template>
