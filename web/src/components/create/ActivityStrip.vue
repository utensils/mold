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
import {
  mergeActivity,
  type ActivityAction,
  type ActivityJobVM,
} from "@studio/lib/activity";
import type { Job } from "../../composables/useGenerateStream";
import { ORIGIN_HOST_ID } from "../../lib/hostRegistry";

const props = withDefaults(
  defineProps<{
    jobs: Job[];
    /** Durable sequence jobs merged into the same strip (mockup 1c: the
     * chain Jobs list merges with the activity strip). */
    sequences?: ActivityJobVM[];
  }>(),
  { sequences: () => [] },
);

/** Which machine a job is running on — omitted when it's this server, so the
 * single-host case stays uncluttered and a routed job is always attributed. */
function hostBadge(job: Job): string | null {
  if (!job.hostId || job.hostId === ORIGIN_HOST_ID) return null;
  return job.hostLabel ?? job.hostId;
}

const emit = defineEmits<{
  cancel: [id: string];
  dismiss: [id: string];
  open: [job: Job];
  "sequence-action": [action: ActivityAction, vm: ActivityJobVM];
  "clear-inactive": [];
  "cleanup-disk": [];
}>();

// mergeActivity orders active work first (running before queued), then by
// recency — the shared rule for both prints and sequences.
const orderedSequences = computed(() =>
  mergeActivity([], props.sequences).filter((vm) => vm.kind === "sequence"),
);

const ACTION_LABELS: Record<ActivityAction, string> = {
  watch: "Watch",
  cancel: "Cancel",
  retake: "Retake",
  edit: "Edit",
  resume: "Resume",
  delete: "Delete",
};

function sequenceHostBadge(vm: ActivityJobVM): string | null {
  return vm.hostId === ORIGIN_HOST_ID ? null : vm.hostLabel;
}

function sequencePercent(vm: ActivityJobVM): number | null {
  if (!vm.progress?.total) return null;
  return Math.round((vm.progress.step / vm.progress.total) * 100);
}

function sequenceStageLabel(vm: ActivityJobVM): string | null {
  if (vm.kind !== "sequence") return null;
  if (vm.state !== "running" && vm.state !== "queued") return null;
  const clip = Math.min(vm.currentStage + 1, vm.stageCount);
  return `clip ${clip}/${vm.stageCount}`;
}

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
const errors = computed(() =>
  props.jobs.filter((j) => j.state === "error" && j.error),
);
const active = computed(
  () =>
    running.value.length +
      queued.value.length +
      errors.value.length +
      props.sequences.length >
    0,
);
</script>

<template>
  <div v-if="active" class="activity" data-test="activity-strip">
    <div class="activity__head">
      <div class="activity__kicker">Activity</div>
      <div v-if="sequences.length" class="activity__head-actions">
        <button
          type="button"
          class="activity__head-action"
          data-test="activity-clear-inactive"
          @click="emit('clear-inactive')"
        >
          Clear inactive
        </button>
        <button
          type="button"
          class="activity__head-action"
          data-test="activity-cleanup-disk"
          @click="emit('cleanup-disk')"
        >
          Clean up disk
        </button>
      </div>
    </div>

    <div v-for="job in running" :key="job.id" class="activity__running">
      <button
        type="button"
        class="activity__open"
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
      <button
        type="button"
        class="activity__cancel activity__cancel--running"
        :aria-label="`Cancel ${promptFor(job)}`"
        :data-test="`activity-cancel-${job.id}`"
        @click="emit('cancel', job.id)"
      >
        <Icon name="close" :size="14" />
      </button>
    </div>

    <div
      v-for="vm in orderedSequences"
      :key="vm.key"
      class="activity__sequence"
      :data-test="`activity-sequence-${vm.kind === 'sequence' ? vm.jobId : ''}`"
    >
      <span class="activity__seq-icon" aria-hidden="true">
        <Icon name="video" :size="14" />
      </span>
      <span class="activity__seq-body">
        <span class="activity__prompt">
          <span
            v-if="sequenceHostBadge(vm)"
            class="activity__host"
            :data-test="`activity-sequence-host-${vm.kind === 'sequence' ? vm.jobId : ''}`"
            >{{ sequenceHostBadge(vm) }}</span
          >
          {{ vm.model }}
          <span v-if="vm.kind === 'sequence'" class="activity__seq-meta">
            · {{ vm.stageCount }} clips · {{ vm.state }}
            <template v-if="sequenceStageLabel(vm)">
              · {{ sequenceStageLabel(vm) }}</template
            >
          </span>
        </span>
        <ProgressBar
          v-if="sequencePercent(vm) !== null"
          :value="sequencePercent(vm) ?? 0"
          tone="accent"
          :height="3"
          :label="`${vm.model} sequence progress`"
        />
        <span
          v-if="vm.kind === 'sequence' && vm.error"
          class="activity__seq-error"
          role="alert"
          >{{ vm.error }}</span
        >
      </span>
      <span v-if="sequencePercent(vm) !== null" class="activity__pct"
        >{{ sequencePercent(vm) }}%</span
      >
      <span class="activity__seq-actions">
        <button
          v-for="action in vm.actions"
          :key="action"
          type="button"
          class="activity__seq-action"
          :data-action="action"
          @click="emit('sequence-action', action, vm)"
        >
          {{ ACTION_LABELS[action] }}
        </button>
      </span>
    </div>

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

    <div
      v-for="job in errors"
      :key="job.id"
      class="activity__error"
      role="alert"
      :data-test="`activity-error-${job.id}`"
    >
      <Icon name="negative" :size="15" />
      <span class="activity__error-body">
        <span class="activity__error-prompt">{{ promptFor(job) }}</span>
        <span>{{ job.error }}</span>
      </span>
      <button
        type="button"
        class="activity__dismiss"
        :aria-label="`Dismiss error for ${promptFor(job)}`"
        :data-test="`activity-dismiss-${job.id}`"
        @click="emit('dismiss', job.id)"
      >
        <Icon name="close" :size="13" />
      </button>
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

.activity__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.activity__head-actions {
  display: flex;
  gap: 6px;
}

.activity__head-action {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-3);
  padding: 3px 9px;
  border-radius: var(--radius-pill);
  font-family: var(--f-mono);
  font-size: 10px;
  cursor: pointer;
}
.activity__head-action:hover {
  border-color: var(--safelight);
  color: var(--rebate);
}

.activity__sequence {
  display: flex;
  align-items: center;
  gap: 11px;
  border: 1px solid var(--edge);
  background: var(--bench);
  border-radius: var(--radius-control);
  padding: 9px 12px;
}

.activity__seq-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  flex: 0 0 26px;
  border-radius: 6px;
  background: var(--bath);
  color: var(--ink-3);
}

.activity__seq-body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.activity__seq-meta {
  font-family: var(--f-mono);
  font-size: 10.5px;
  color: var(--ink-3);
}

.activity__seq-error {
  font-size: 11px;
  color: var(--stop);
  overflow-wrap: anywhere;
}

.activity__seq-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  flex: 0 0 auto;
}

.activity__seq-action {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 3px 9px;
  border-radius: var(--radius-control);
  font-family: var(--f-mono);
  font-size: 10.5px;
  cursor: pointer;
}
.activity__seq-action:hover {
  border-color: var(--safelight);
  color: var(--rebate);
}

.activity__running {
  display: flex;
  align-items: center;
  width: 100%;
  border: 1px solid var(--edge);
  background: var(--bench);
  border-radius: var(--radius-control);
  padding: 0;
}

.activity__open {
  display: flex;
  align-items: center;
  gap: 11px;
  flex: 1;
  min-width: 0;
  border: 0;
  background: transparent;
  padding: 9px 12px;
  color: inherit;
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

.activity__error {
  display: flex;
  align-items: flex-start;
  gap: 9px;
  border: 1px solid color-mix(in srgb, var(--stop) 45%, var(--edge));
  background: color-mix(in srgb, var(--stop) 10%, var(--bench));
  border-radius: var(--radius-control);
  padding: 10px 12px;
  color: var(--stop);
  font-size: 12px;
}

.activity__error-body {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
  gap: 3px;
  overflow-wrap: anywhere;
}

.activity__error-prompt {
  color: var(--ink-2);
  font-weight: 600;
}

.activity__dismiss {
  flex: 0 0 auto;
  border-radius: 4px;
  padding: 2px;
  color: var(--ink-3);
}

.activity__dismiss:hover {
  background: color-mix(in srgb, var(--stop) 15%, transparent);
  color: var(--stop);
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

.activity__cancel--running {
  min-width: 44px;
  min-height: 44px;
  border-left: 1px solid var(--edge);
}
</style>
