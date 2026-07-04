<script setup lang="ts">
import { computed, onMounted, ref, toRef } from "vue";
import { getChainJob, listChainJobs } from "../api";
import { useChainJobStream } from "../composables/useChainJobStream";
import type { ChainJobDetail, ChainJobSummary } from "../types";
import ChainJobCard from "./ChainJobCard.vue";

const props = withDefaults(
  defineProps<{
    jobs?: ChainJobDetail[];
  }>(),
  {
    jobs: () => [],
  },
);

const summaries = ref<ChainJobSummary[]>([]);
const selectedId = ref<string | null>(null);
const selectedFallback = ref<ChainJobDetail | null>(null);
const streamed = useChainJobStream(selectedId);

async function load() {
  if (props.jobs.length > 0) return;
  const listing = await listChainJobs();
  summaries.value = listing.jobs;
  selectedId.value = listing.jobs[0]?.id ?? null;
  if (selectedId.value) {
    selectedFallback.value = await getChainJob(selectedId.value);
  }
}

const renderedJobs = computed(() => {
  if (props.jobs.length > 0) return props.jobs;
  return [streamed.detail.value ?? selectedFallback.value].filter(
    (job): job is ChainJobDetail => Boolean(job),
  );
});

onMounted(() => {
  void load();
});

const jobsRef = toRef(props, "jobs");
</script>

<template>
  <section data-test="jobs-panel" class="flex flex-col gap-2">
    <div
      v-if="jobsRef.length === 0 && summaries.length > 1"
      class="flex flex-wrap gap-2"
      data-test="jobs-panel-tabs"
    >
      <button
        v-for="job in summaries"
        :key="job.id"
        class="rounded border border-slate-700 px-2 py-1 text-xs text-slate-200"
        :class="{ 'bg-slate-800': job.id === selectedId }"
        @click="selectedId = job.id"
      >
        {{ job.id }}
      </button>
    </div>
    <ChainJobCard
      v-for="job in renderedJobs"
      :key="job.id"
      :job="job"
      @updated="load"
    />
  </section>
</template>
