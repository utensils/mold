<script setup lang="ts">
import type { ChainJobDetail } from "../types";

defineProps<{
  job: ChainJobDetail;
}>();
</script>

<template>
  <article
    data-test="chain-job-card"
    class="rounded-lg border border-slate-700 p-3"
  >
    <!-- TODO(BR55 Phase 6): Render durable chain job stage timeline with preview thumbnails from chainJobStagePreviewUrl. -->
    <!-- TODO(BR55 Phase 6): Wire Resume Cancel and Retake controls including disabled splice tooltip for smooth next transition. -->
    <div class="text-sm font-medium text-slate-100">{{ job.id }}</div>
    <div class="text-xs text-slate-400">{{ job.state }}</div>
  </article>
</template>
