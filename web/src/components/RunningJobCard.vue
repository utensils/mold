<script setup lang="ts">
import { computed } from "vue";
import type { Job } from "../composables/useGenerateStream";

const props = defineProps<{ job: Job }>();
const emit = defineEmits<{ (e: "cancel", id: string): void }>();

const pct = computed(() => {
  const p = props.job.progress;
  if (p.step !== null && p.totalSteps) {
    return Math.round((p.step / p.totalSteps) * 100);
  }
  if (p.weightBytesLoaded !== null && p.weightBytesTotal) {
    return Math.round((p.weightBytesLoaded / p.weightBytesTotal) * 100);
  }
  return null;
});

const thumbSrc = computed(() => {
  const r = props.job.result;
  if (!r) return null;
  const poster = r.video_thumbnail ?? r.image;
  return `data:image/*;base64,${poster}`;
});
</script>

<template>
  <div
    class="glass flex w-[280px] flex-shrink-0 flex-col gap-2 rounded-2xl p-3"
  >
    <div class="flex items-center justify-between text-xs text-slate-400">
      <span>{{ job.request.model }}</span>
      <span v-if="job.progress.gpu !== null">GPU {{ job.progress.gpu }}</span>
    </div>
    <div
      class="relative aspect-square overflow-hidden rounded-xl bg-slate-900/60"
    >
      <img
        v-if="thumbSrc"
        :src="thumbSrc"
        class="h-full w-full object-cover"
        alt=""
      />
      <div v-else class="h-full w-full animate-pulse bg-slate-800/60"></div>
      <div
        v-if="job.state === 'error'"
        class="absolute inset-0 flex items-center justify-center bg-rose-500/70 p-2 text-center text-xs text-white"
      >
        {{ job.error }}
      </div>
    </div>
    <div class="text-xs text-slate-300">{{ job.progress.stage }}</div>
    <div
      v-if="pct !== null"
      class="h-1 w-full overflow-hidden rounded-full bg-slate-900/60"
    >
      <div
        class="h-full bg-brand-500 transition-all"
        :style="{ width: pct + '%' }"
      ></div>
    </div>
    <div class="flex justify-between text-xs text-slate-500">
      <span v-if="job.progress.step !== null"
        >{{ job.progress.step }} / {{ job.progress.totalSteps }}</span
      >
      <span v-else>&nbsp;</span>
      <button
        v-if="job.state === 'running'"
        type="button"
        class="text-slate-400 hover:text-rose-300"
        @click="emit('cancel', job.id)"
      >
        ✕
      </button>
    </div>
  </div>
</template>
