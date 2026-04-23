<script setup lang="ts">
import { computed } from "vue";
import type { Job } from "../composables/useGenerateStream";

// Hide-mode renders the thumbnail behind a blurred shroud until the user
// reveals it. `revealed` is a per-card boolean; the parent tracks the
// global peek set and looks up by job.id.
const props = withDefaults(
  defineProps<{
    job: Job;
    hideMode?: boolean;
    revealed?: boolean;
  }>(),
  {
    hideMode: false,
    revealed: false,
  },
);
const emit = defineEmits<{
  (e: "cancel", id: string): void;
  (e: "open", job: Job): void;
  (e: "dismiss", id: string): void;
  (e: "reveal", id: string): void;
}>();

const isHidden = computed(() => props.hideMode && !props.revealed);

function onReveal(evt: Event) {
  evt.stopPropagation();
  emit("reveal", props.job.id);
}

// Done jobs are clickable — they open the gallery detail drawer for the
// saved file. The parent does the Job→GalleryImage lookup since the SSE
// complete event doesn't echo the on-disk filename.
const clickable = computed(
  () => props.job.state === "done" && props.job.result !== null,
);

function onClick() {
  // Shrouded cards should not leak the finished image through the detail
  // drawer — the user must reveal first.
  if (isHidden.value) return;
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
      <!-- Hide shroud. Matches the gallery card: heavy blur + dim, with
           a Reveal button that peeks this job without flipping the global
           toggle. Stop-propagates so the parent's open handler doesn't fire. -->
      <div
        v-if="isHidden"
        class="absolute inset-0 z-[5] flex flex-col items-center justify-center gap-2 bg-slate-950/70 text-slate-100 backdrop-blur-2xl"
      >
        <svg
          class="h-5 w-5 text-slate-300"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="1.8"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <path
            d="M10.6 5.1A10 10 0 0 1 12 5c6 0 10 7 10 7a17 17 0 0 1-3.3 4.2"
          />
          <path d="M6.7 6.7A17 17 0 0 0 2 12s4 7 10 7a9.7 9.7 0 0 0 5.3-1.7" />
          <path d="m3 3 18 18" />
          <path d="M9.9 9.9a3 3 0 0 0 4.2 4.2" />
        </svg>
        <button
          type="button"
          class="rounded-full bg-white/10 px-3 py-1 text-[12px] font-medium text-slate-100 transition hover:bg-white/20"
          @click="onReveal"
          @keydown.stop
        >
          Reveal
        </button>
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
        :aria-label="'Cancel job'"
        @click.stop="emit('cancel', job.id)"
      >
        ✕
      </button>
      <button
        v-else
        type="button"
        class="text-slate-500 hover:text-slate-200"
        :aria-label="'Dismiss card'"
        @click.stop="emit('dismiss', job.id)"
      >
        ✕
      </button>
    </div>
  </div>
</template>
