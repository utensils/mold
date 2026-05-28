<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from "vue";
import { STALE_THRESHOLD_MS, type Job } from "../composables/useGenerateStream";
import type { QueueEntry } from "../types";

const props = withDefaults(
  defineProps<{
    job: Job;
    queueEntry?: QueueEntry | null;
    gpus?: Array<{ ordinal: number; state?: string }>;
  }>(),
  {
    queueEntry: null,
    gpus: () => [],
  },
);
const emit = defineEmits<{
  (e: "cancel", id: string): void;
  (e: "open", job: Job): void;
  (e: "dismiss", id: string): void;
  (e: "lane-change", id: string, targetGpu: number | null): void;
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

// Reactive wall-clock ticked once per 10 s. Powers the `isStale` predicate
// without re-rendering 60 fps when nothing actually changed. Stops on
// unmount so this card doesn't keep a timer alive after the running strip
// scrolls it off.
const now = ref(Date.now());
const tickId = window.setInterval(() => {
  now.value = Date.now();
}, 10_000);
onBeforeUnmount(() => window.clearInterval(tickId));

// Stale = `running` but no SSE progress event has landed within
// STALE_THRESHOLD_MS after the job has moved past the queued phase.
// Queued jobs can sit silently for a long time; only actual work going
// quiet is suspicious. The L1 silent-close fix in api.ts catches most
// dropped streams within the SSE keepalive window (15 s), so by the
// time we surface this badge the connection is almost certainly dead.
const isStale = computed(() => {
  if (props.job.state !== "running") return false;
  if (!props.job.workStarted) return false;
  return now.value - props.job.lastProgressAt > STALE_THRESHOLD_MS;
});

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

const laneValue = computed(() => {
  const target =
    props.queueEntry?.target_gpu ?? props.queueEntry?.preferred_gpu;
  return target === null || target === undefined ? "" : String(target);
});

const laneSelectDisabled = computed(
  () => !props.queueEntry || props.queueEntry.state === "running",
);

function onLaneChange(evt: Event) {
  if (!props.queueEntry || laneSelectDisabled.value) return;
  const value = (evt.target as HTMLSelectElement).value;
  emit(
    "lane-change",
    props.queueEntry.id,
    value === "" ? null : Number.parseInt(value, 10),
  );
}
</script>

<template>
  <div
    data-test="running-card"
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
    <label
      v-if="queueEntry"
      class="flex items-center justify-between gap-2 text-[11px] text-slate-400"
    >
      <span>Lane</span>
      <select
        data-test="job-lane-select"
        class="rounded border border-white/10 bg-slate-950 px-2 py-1 text-xs text-slate-200 disabled:opacity-50"
        :value="laneValue"
        :disabled="laneSelectDisabled"
        @change="onLaneChange"
      >
        <option value="">Auto</option>
        <option
          v-for="gpu in gpus"
          :key="gpu.ordinal"
          :value="String(gpu.ordinal)"
        >
          GPU {{ gpu.ordinal }}
        </option>
      </select>
    </label>
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
      v-if="isStale"
      role="status"
      class="flex items-center gap-1 text-[11px] text-amber-400"
    >
      <span class="inline-block h-1.5 w-1.5 rounded-full bg-amber-400"></span>
      <span>
        No progress for &gt;{{ Math.floor(STALE_THRESHOLD_MS / 1000) }}s —
        stream may have dropped. Cancel and retry if needed.
      </span>
    </div>
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
