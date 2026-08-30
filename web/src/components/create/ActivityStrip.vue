<script setup lang="ts">
/*
 * Activity strip (Mold Studio Create) — the "Activity" row above the composer,
 * and it is present tense. Active jobs show a shimmer thumb + prompt + mono
 * percent + a thin progress bar; jobs still waiting in the queue show as pills
 * with an ✕ cancel. Settled-but-wrong work (a failed print, a failed or
 * interrupted sequence) keeps a dismissible row for five minutes, capped at
 * two; everything else settled collapses into one digest chip that opens
 * Library ▸ History ▸ Sequences. Hidden entirely when nothing is in flight and
 * there is nothing left to count. The per-GPU lane view lives in host detail.
 */
import { computed, reactive } from "vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import Icon from "@ui/components/Icon.vue";
import LiveActivityList from "@ui/components/LiveActivityList.vue";
import type { FleetActiveWork } from "@studio/api/activity";
import {
  activityDigestLabel,
  mergeActivity,
  partitionActivity,
  type ActivityAction,
  type ActivityJobVM,
} from "@studio/lib/activity";
import {
  queueStatusFor,
  queueWaitLabel,
  resolveQueueWait,
  type QueueStatusIndex,
} from "@studio/lib/queuePosition";
import type { Job } from "../../composables/useGenerateStream";
import { ORIGIN_HOST_ID } from "../../lib/hostRegistry";
import { compareNewestSubmitted } from "@studio/lib/activityOrder";

const props = withDefaults(
  defineProps<{
    jobs: Job[];
    /** Durable sequence jobs merged into the same strip (mockup 1c: the
     * chain Jobs list merges with the activity strip). */
    sequences?: ActivityJobVM[];
    /** Server-owned work discovered after a reload or in another client. */
    shared?: FleetActiveWork[];
    /** Live dispatch order per host from `/api/queue`. Absent (or missing an
     * entry) simply means the pill says "Queued" and nothing more. */
    queueStatus?: QueueStatusIndex | null;
  }>(),
  { sequences: () => [], shared: () => [], queueStatus: null },
);

/** "Next up" / "#2 in line", or "Waiting for memory" when the scheduler really
 * did park the job. Resolved in the shared studio layer so web, desktop, and
 * iPhone describe the same waiting row the same way; a server that lists
 * nothing degrades to the plain "Queued" pill. */
function queueLabel(job: Job): string {
  if (job.holdError) return job.progress.stage;
  if (job.detached && job.durableBatch && !job.serverId) {
    return "Confirming durable admission";
  }
  return queueWaitLabel(
    resolveQueueWait(
      queueStatusFor(
        props.queueStatus,
        job.hostId ?? ORIGIN_HOST_ID,
        job.serverId,
      ),
    ),
  );
}

/** Which machine a job is running on — omitted when it's this server, so the
 * single-host case stays uncluttered and a routed job is always attributed. */
function hostBadge(job: Job): string | null {
  if (!job.hostId || job.hostId === ORIGIN_HOST_ID) return null;
  return job.hostLabel ?? job.hostId;
}

const emit = defineEmits<{
  cancel: [id: string];
  retry: [id: string];
  dismiss: [id: string];
  open: [job: Job];
  "sequence-action": [action: ActivityAction, vm: ActivityJobVM];
  "show-history": [];
  "shared-open": [row: FleetActiveWork];
}>();

/** Session-only sequence dismissals, keyed by VM key. Deliberately not
 * persisted: the age rule already reads server timestamps and survives a
 * reload, and a second retention mechanism could disagree with it. */
const dismissedSequences = reactive(new Set<string>());

/** Failed prints join the same partition so their rows expire on the same
 * clock as sequences. Running/queued prints keep the strip's own chrome. */
const printVMs = computed<ActivityJobVM[]>(() =>
  props.jobs
    .filter((job) => job.state === "error" && job.error)
    .map((job) => ({
      kind: "print" as const,
      key: `print:${job.id}`,
      hostId: job.hostId ?? ORIGIN_HOST_ID,
      hostLabel: job.hostLabel ?? "",
      model: job.request.model,
      prompt: promptFor(job),
      phase: "failed" as const,
      progress: null,
      chain: null,
      actions: [],
      createdAtMs: job.startedAt,
      settledAtMs: job.settledAt,
      error: job.error,
    })),
);

const partition = computed(() =>
  partitionActivity(mergeActivity(printVMs.value, props.sequences), {
    dismissed: dismissedSequences,
  }),
);

const activeSequences = computed(() =>
  partition.value.active.filter(
    (vm): vm is ActivityJobVM & { kind: "sequence" } => vm.kind === "sequence",
  ),
);
const attentionSequences = computed(() =>
  partition.value.attention.filter(
    (vm): vm is ActivityJobVM & { kind: "sequence" } => vm.kind === "sequence",
  ),
);

/** Which sequence rows carry the non-destructive ✕. */
const attentionKeys = computed(
  () => new Set(partition.value.attention.map((vm) => vm.key)),
);

/** Failed prints the strip is still holding, in partition order. */
const errors = computed(() =>
  partition.value.attention.flatMap((vm) => {
    if (vm.kind !== "print") return [];
    const job = props.jobs.find((j) => `print:${j.id}` === vm.key);
    return job ? [job] : [];
  }),
);

const digest = computed(() => activityDigestLabel(partition.value));

function dismissSequence(vm: ActivityJobVM) {
  dismissedSequences.add(vm.key);
}

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
  return p.step !== null && p.totalSteps
    ? Math.round((p.step / p.totalSteps) * 100)
    : null;
}

function isFinalizing(job: Job): boolean {
  const { stage, step, totalSteps } = job.progress;
  return (
    step !== null &&
    totalSteps !== null &&
    step >= totalSteps &&
    stage !== "Denoising" &&
    stage !== "Developing"
  );
}

/** `title ?? prompt ?? "Untitled print"` — a chain request carries no title
 * slot, so the `in` check keeps the union honest. */
function promptFor(job: Job): string {
  const title =
    "title" in job.request && typeof job.request.title === "string"
      ? job.request.title.trim()
      : "";
  return title || job.request.prompt?.trim() || "Untitled print";
}

function terminalLabel(job: Job): string {
  return job.detached
    ? "Detached — the original machine still owns the outcome"
    : "Failed — open Create for details";
}

const running = computed(() =>
  props.jobs.filter((j) => j.state === "running" && j.workStarted),
);
const queued = computed(() =>
  props.jobs.filter((j) => j.state === "running" && !j.workStarted),
);
/** Render exactly the newest submitted queued print, then summarize the rest.
 * Cancellation reveals the preceding submission without producing one
 * interactive DOM subtree per backlog item. */
const newestQueued = computed(() => {
  const actionable = queued.value.filter((job) => !job.cancelling);
  return (
    [...(actionable.length > 0 ? actionable : queued.value)].sort((a, b) =>
      compareNewestSubmitted(
        { createdAtMs: a.startedAt },
        { createdAtMs: b.startedAt },
      ),
    )[0] ?? null
  );
});
const summarizedQueuedCount = computed(() =>
  Math.max(0, queued.value.length - (newestQueued.value ? 1 : 0)),
);

type WebActivityRow =
  | {
      key: string;
      createdAtMs: number;
      kind: "shared";
      shared: FleetActiveWork;
    }
  | { key: string; createdAtMs: number; kind: "print"; print: Job }
  | {
      key: string;
      createdAtMs: number;
      kind: "sequence";
      sequence: ActivityJobVM & { kind: "sequence" };
    };

/** One newest-first visual queue across local prints, sequences, and recovered
 * fleet work. A phase transition changes only the row contents. */
const activeRows = computed<WebActivityRow[]>(() =>
  [
    ...props.shared.map((shared): WebActivityRow => ({
      key: `shared:${shared.key}`,
      createdAtMs: shared.created_at_unix_ms,
      kind: "shared",
      shared,
    })),
    ...running.value.map((print): WebActivityRow => ({
      key: `print:${print.id}`,
      createdAtMs: print.startedAt,
      kind: "print",
      print,
    })),
    ...(newestQueued.value
      ? [
          {
            key: `print:${newestQueued.value.id}`,
            createdAtMs: newestQueued.value.startedAt,
            kind: "print" as const,
            print: newestQueued.value,
          },
        ]
      : []),
    ...activeSequences.value.map((sequence): WebActivityRow => ({
      key: sequence.key,
      createdAtMs: sequence.createdAtMs,
      kind: "sequence",
      sequence,
    })),
  ].sort(compareNewestSubmitted),
);
const active = computed(
  () =>
    running.value.length > 0 ||
    queued.value.length > 0 ||
    partition.value.active.length > 0 ||
    partition.value.attention.length > 0 ||
    props.shared.length > 0 ||
    digest.value !== null,
);
</script>

<template>
  <div v-if="active" class="activity" data-test="activity-strip">
    <div class="activity__head">
      <div class="activity__kicker">Activity</div>
      <button
        v-if="digest"
        type="button"
        class="activity__digest"
        data-test="activity-digest"
        title="Show settled sequences in History"
        @click="emit('show-history')"
      >
        {{ digest }}
      </button>
    </div>

    <template v-for="row in activeRows" :key="row.key">
      <LiveActivityList
        v-if="row.kind === 'shared'"
        :rows="[row.shared]"
        interactive
        @select="emit('shared-open', $event)"
      />

      <div
        v-else-if="row.kind === 'print' && row.print.workStarted"
        class="activity__running"
      >
        <button
          type="button"
          class="activity__open"
          :data-test="`activity-running-${row.print.id}`"
          @click="emit('open', row.print)"
        >
          <span class="activity__thumb ms-shimmer" aria-hidden="true" />
          <span class="activity__body">
            <span class="activity__prompt">
              <span
                v-if="hostBadge(row.print)"
                class="activity__host"
                :data-test="`activity-host-${row.print.id}`"
                >{{ hostBadge(row.print) }}</span
              >
              {{ promptFor(row.print) }}
            </span>
            <ProgressBar
              :value="percentFor(row.print) ?? 0"
              tone="accent"
              :height="3"
              :label="
                isFinalizing(row.print)
                  ? `${promptFor(row.print)} finalizing`
                  : `${promptFor(row.print)} progress`
              "
            />
          </span>
          <span
            v-if="isFinalizing(row.print)"
            class="activity__pct activity__pct--stage"
            >Finalizing</span
          >
          <span v-else-if="percentFor(row.print) !== null" class="activity__pct"
            >{{ percentFor(row.print) }}%</span
          >
          <span v-else class="activity__pct activity__pct--stage">{{
            row.print.progress.stage
          }}</span>
        </button>
        <button
          type="button"
          class="activity__cancel activity__cancel--running"
          :aria-label="
            row.print.cancelling
              ? `Cancelling ${promptFor(row.print)}`
              : `Cancel ${promptFor(row.print)}`
          "
          :data-test="`activity-cancel-${row.print.id}`"
          :disabled="row.print.cancelling"
          @click="emit('cancel', row.print.id)"
        >
          <span v-if="row.print.cancelling" class="data-mono">…</span>
          <Icon v-else name="close" :size="14" />
        </button>
      </div>

      <div
        v-else-if="row.kind === 'sequence'"
        class="activity__sequence"
        :data-test="`activity-sequence-${row.sequence.jobId}`"
        role="button"
        tabindex="0"
        @click="emit('sequence-action', 'watch', row.sequence)"
        @keydown.enter.prevent="emit('sequence-action', 'watch', row.sequence)"
        @keydown.space.prevent="emit('sequence-action', 'watch', row.sequence)"
      >
        <span class="activity__seq-icon" aria-hidden="true">
          <Icon name="video" :size="14" />
        </span>
        <span class="activity__seq-body">
          <span class="activity__prompt">
            <span
              v-if="sequenceHostBadge(row.sequence)"
              class="activity__host"
              :data-test="`activity-sequence-host-${row.sequence.jobId}`"
              >{{ sequenceHostBadge(row.sequence) }}</span
            >
            {{ row.sequence.model }}
            <span class="activity__seq-meta">
              · {{ row.sequence.stageCount }} clips ·
              {{ row.sequence.phase ?? row.sequence.state }}
              <template v-if="sequenceStageLabel(row.sequence)">
                · {{ sequenceStageLabel(row.sequence) }}</template
              >
            </span>
          </span>
          <ProgressBar
            v-if="sequencePercent(row.sequence) !== null"
            :value="sequencePercent(row.sequence) ?? 0"
            tone="accent"
            :height="3"
            :label="`${row.sequence.model} sequence progress`"
          />
        </span>
        <span
          v-if="sequencePercent(row.sequence) !== null"
          class="activity__pct"
          >{{ sequencePercent(row.sequence) }}%</span
        >
        <span class="activity__seq-actions">
          <button
            v-for="action in row.sequence.actions"
            :key="action"
            type="button"
            class="activity__seq-action"
            :data-action="action"
            @click.stop="emit('sequence-action', action, row.sequence)"
          >
            {{ ACTION_LABELS[action] }}
          </button>
        </span>
      </div>

      <div v-else class="activity__queued">
        <span
          class="activity__pill"
          :data-test="`activity-queued-${row.print.id}`"
          role="button"
          tabindex="0"
          @click="emit('open', row.print)"
          @keydown.enter.prevent="emit('open', row.print)"
          @keydown.space.prevent="emit('open', row.print)"
        >
          <span class="activity__pill-text">
            <span
              v-if="hostBadge(row.print)"
              class="activity__host"
              :data-test="`activity-host-${row.print.id}`"
              >{{ hostBadge(row.print) }}</span
            >
            <span
              class="activity__queue-position"
              :data-test="`activity-queue-position-${row.print.id}`"
              >{{ queueLabel(row.print) }}</span
            >
            {{ promptFor(row.print) }}
            <span v-if="row.print.holdError" class="activity__hold-error">
              · {{ row.print.holdError }}
            </span>
          </span>
          <button
            v-if="row.print.retryable"
            type="button"
            class="activity__seq-action"
            :disabled="row.print.retrying"
            :data-test="`activity-retry-${row.print.id}`"
            @click.stop="emit('retry', row.print.id)"
          >
            {{ row.print.retrying ? "Retrying…" : "Retry" }}
          </button>
          <button
            type="button"
            class="activity__cancel"
            :aria-label="`Cancel ${promptFor(row.print)}`"
            :data-test="`activity-cancel-${row.print.id}`"
            :disabled="row.print.cancelling"
            @click.stop="emit('cancel', row.print.id)"
          >
            <span v-if="row.print.cancelling" class="data-mono">…</span>
            <Icon v-else name="close" :size="12" />
          </button>
        </span>
      </div>
    </template>

    <span
      v-if="summarizedQueuedCount"
      class="activity__queue-summary"
      data-test="activity-queued-summary"
    >
      {{ summarizedQueuedCount }} other queued
      {{ summarizedQueuedCount === 1 ? "print" : "prints" }}
    </span>

    <div
      v-for="vm in attentionSequences"
      :key="vm.key"
      class="activity__sequence"
      :data-test="`activity-sequence-${vm.jobId}`"
      role="button"
      tabindex="0"
      @click="emit('sequence-action', 'watch', vm)"
      @keydown.enter.prevent="emit('sequence-action', 'watch', vm)"
      @keydown.space.prevent="emit('sequence-action', 'watch', vm)"
    >
      <span class="activity__seq-icon" aria-hidden="true">
        <Icon name="video" :size="14" />
      </span>
      <span class="activity__seq-body">
        <span class="activity__prompt">{{ vm.model }}</span>
      </span>
      <span class="activity__seq-actions">
        <button
          v-for="action in vm.actions"
          :key="action"
          type="button"
          class="activity__seq-action"
          :data-action="action"
          @click.stop="emit('sequence-action', action, vm)"
        >
          {{ ACTION_LABELS[action] }}
        </button>
        <button
          v-if="attentionKeys.has(vm.key)"
          type="button"
          class="activity__cancel"
          :data-test="`activity-seq-dismiss-${vm.jobId}`"
          title="Hide this from Activity. The sequence stays in Library ▸ History."
          :aria-label="`Dismiss ${vm.model}`"
          @click.stop="dismissSequence(vm)"
        >
          <Icon name="close" :size="13" />
        </button>
      </span>
    </div>

    <div
      v-for="job in errors"
      :key="job.id"
      class="activity__error"
      role="alert"
      :data-test="`activity-error-${job.id}`"
      tabindex="0"
      @click="emit('open', job)"
      @keydown.enter.prevent="emit('open', job)"
      @keydown.space.prevent="emit('open', job)"
    >
      <span class="activity__error-body">
        <span class="activity__error-prompt">{{ promptFor(job) }}</span>
        <span>{{ terminalLabel(job) }}</span>
      </span>
      <button
        type="button"
        class="activity__dismiss"
        :aria-label="`Dismiss error for ${promptFor(job)}`"
        :data-test="`activity-dismiss-${job.id}`"
        @click.stop="emit('dismiss', job.id)"
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

.activity__digest {
  border: 0;
  background: transparent;
  color: var(--ink-3);
  padding: 0;
  font-family: var(--f-mono);
  font-size: 9.5px;
  letter-spacing: 0.04em;
  cursor: pointer;
}
.activity__digest:hover {
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

.activity__queue-position {
  display: inline-block;
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.06em;
  color: var(--safelight);
  margin-right: 5px;
  vertical-align: middle;
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

.activity__queue-summary {
  align-self: center;
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
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
