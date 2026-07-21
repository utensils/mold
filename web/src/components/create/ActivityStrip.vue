<script setup lang="ts">
/*
 * Activity strip (Mold Studio Create) — the "Activity" row above the composer.
 * Active jobs show a shimmer thumb + prompt + mono percent + a thin progress
 * bar; jobs still waiting in the queue show as pills with an ✕ cancel. Hidden
 * entirely when nothing is in flight. The per-GPU lane view lives in host
 * detail now — this strip is the lightweight at-a-glance indicator.
 */
import { computed } from "vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import Icon from "@ui/components/Icon.vue";
import type { Job } from "../../composables/useGenerateStream";
import { ORIGIN_HOST_ID } from "../../lib/hostRegistry";

const props = defineProps<{ jobs: Job[] }>();

/** Which machine a job is running on — omitted when it's this server, so the
 * single-host case stays uncluttered and a routed job is always attributed. */
function hostBadge(job: Job): string | null {
  if (!job.hostId || job.hostId === ORIGIN_HOST_ID) return null;
  return job.hostLabel ?? job.hostId;
}

const emit = defineEmits<{
  cancel: [id: string];
  open: [job: Job];
}>();

function percentFor(job: Job): number | null {
  const p = job.progress;
  if (p.step !== null && p.totalSteps) {
    return Math.round((p.step / p.totalSteps) * 100);
  }
  if (p.weightBytesLoaded !== null && p.weightBytesTotal) {
    return Math.round((p.weightBytesLoaded / p.weightBytesTotal) * 100);
  }
  return null;
}

function promptFor(job: Job): string {
  return job.request.prompt?.trim() || "Untitled print";
}

const running = computed(() =>
  props.jobs.filter((j) => j.state === "running" && j.workStarted),
);
const queued = computed(() =>
  props.jobs.filter((j) => j.state === "running" && !j.workStarted),
);
const active = computed(() => running.value.length + queued.value.length > 0);
</script>

<template>
  <div v-if="active" class="activity" data-test="activity-strip">
    <div class="activity__kicker">Activity</div>

    <button
      v-for="job in running"
      :key="job.id"
      type="button"
      class="activity__running"
      :data-test="`activity-running-${job.id}`"
      @click="emit('open', job)"
    >
      <span class="activity__thumb ms-shimmer" aria-hidden="true" />
      <span class="activity__body">
        <span class="activity__prompt">
          <span
            v-if="hostBadge(job)"
            class="activity__host"
            :data-test="`activity-host-${job.id}`"
            >{{ hostBadge(job) }}</span
          >
          {{ promptFor(job) }}
        </span>
        <ProgressBar
          :value="percentFor(job) ?? 0"
          tone="accent"
          :height="3"
          :label="`${promptFor(job)} progress`"
        />
      </span>
      <span v-if="percentFor(job) !== null" class="activity__pct"
        >{{ percentFor(job) }}%</span
      >
      <span v-else class="activity__pct activity__pct--stage">{{
        job.progress.stage
      }}</span>
    </button>

    <div v-if="queued.length" class="activity__queued">
      <span
        v-for="job in queued"
        :key="job.id"
        class="activity__pill"
        :data-test="`activity-queued-${job.id}`"
      >
        <span class="activity__pill-text">
          <span
            v-if="hostBadge(job)"
            class="activity__host"
            :data-test="`activity-host-${job.id}`"
            >{{ hostBadge(job) }}</span
          >
          {{ promptFor(job) }}
        </span>
        <button
          type="button"
          class="activity__cancel"
          :aria-label="`Cancel ${promptFor(job)}`"
          :data-test="`activity-cancel-${job.id}`"
          @click="emit('cancel', job.id)"
        >
          <Icon name="close" :size="12" />
        </button>
      </span>
    </div>
  </div>
</template>

<style scoped>
.activity {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 12px;
}

.activity__kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.activity__running {
  display: flex;
  align-items: center;
  gap: 11px;
  width: 100%;
  border: 1px solid var(--edge);
  background: var(--bench);
  border-radius: var(--radius-control);
  padding: 9px 12px;
  text-align: left;
  cursor: pointer;
}

.activity__thumb {
  width: 26px;
  height: 26px;
  flex: 0 0 26px;
  border-radius: 6px;
}

.activity__body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.activity__prompt {
  font-size: 12px;
  color: var(--ink-2);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.activity__host {
  display: inline-block;
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--halide);
  border: 1px solid var(--edge);
  border-radius: var(--radius-pill);
  padding: 1px 6px;
  margin-right: 5px;
  vertical-align: middle;
}

.activity__pct {
  font-family: var(--f-mono);
  font-size: 11px;
  color: var(--safelight);
  flex: 0 0 auto;
}

.activity__pct--stage {
  color: var(--ink-3);
}

.activity__queued {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
}

.activity__pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  max-width: 240px;
  border: 1px solid var(--edge);
  background: var(--bath);
  border-radius: var(--radius-pill);
  padding: 5px 8px 5px 12px;
}

.activity__pill-text {
  font-size: 11.5px;
  color: var(--ink-2);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.activity__cancel {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  background: transparent;
  color: var(--ink-3);
  padding: 2px;
  cursor: pointer;
}

.activity__cancel:hover {
  color: var(--stop);
}
</style>
