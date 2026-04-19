<script setup lang="ts">
import { computed } from "vue";
import type { Job } from "../composables/useGenerateStream";

const props = defineProps<{ job: Job }>();
const emit = defineEmits<{
  (e: "cancel", id: string): void;
  (e: "open", job: Job): void;
}>();

// Done jobs are clickable — they open the gallery detail drawer for the
// saved file. The parent does the Job→GalleryImage lookup since the SSE
// complete event doesn't echo the on-disk filename.
const clickable = computed(
  () => props.job.state === "done" && props.job.result !== null,
);

function onClick() {
  if (clickable.value) emit("open", props.job);
}

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
  // Video: thumbnail is always PNG (server-side). Image: use the declared format.
  if (r.video_thumbnail) return `data:image/png;base64,${r.video_thumbnail}`;
  const mime = r.format === "jpeg" ? "image/jpeg" : `image/${r.format}`;
  return `data:${mime};base64,${r.image}`;
});
</script>

<template>
  <div
    class="glass flex w-[280px] flex-shrink-0 flex-col gap-2 rounded-2xl p-3"
    :class="
      clickable
        ? 'cursor-zoom-in transition hover:bg-white/[0.04] focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-400'
        : ''
    "
    :role="clickable ? 'button' : undefined"
    :tabindex="clickable ? 0 : undefined"
    :aria-label="clickable ? 'Open in detail view' : undefined"
    @click="onClick"
    @keydown.enter.prevent="onClick"
    @keydown.space.prevent="onClick"
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
        @click.stop="emit('cancel', job.id)"
      >
        ✕
      </button>
    </div>
  </div>
</template>
