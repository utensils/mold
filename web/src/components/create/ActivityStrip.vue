<script setup lang="ts">
/*
 * Activity strip (Mold Studio Create) — the "Activity" row above the composer,
 * and it is present tense. Active jobs show a shimmer thumb + prompt + mono
 * percent + a thin progress bar; jobs still waiting in the queue show as pills
 * with an ✕ cancel. A failed print keeps a dismissible row for five minutes,
 * capped at two. Hidden entirely when nothing is in flight and nothing failed
 * recently. The per-GPU lane view lives in host detail.
 */
import { computed } from "vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import Icon from "@ui/components/Icon.vue";
import LiveActivityList from "@ui/components/LiveActivityList.vue";
import type { FleetActiveWork } from "@studio/api/activity";
import {
  mergeActivity,
  partitionActivity,
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
    /** Server-owned work discovered after a reload or in another client. */
    shared?: FleetActiveWork[];
    /** Live dispatch order per host from `/api/queue`. Absent (or missing an
     * entry) simply means the pill says "Queued" and nothing more. */
    queueStatus?: QueueStatusIndex | null;
  }>(),
  { shared: () => [], queueStatus: null },
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
  "shared-open": [row: FleetActiveWork];
}>();

/** Failed prints go through the shared partition so their rows expire on the
 * shared clock. Running/queued prints keep the strip's own chrome. */
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
  partitionActivity(mergeActivity(printVMs.value, [])),
);

/** Failed prints the strip is still holding, in partition order. */
const errors = computed(() =>
  partition.value.attention.flatMap((vm) => {
    if (vm.kind !== "print") return [];
    const job = props.jobs.find((j) => `print:${j.id}` === vm.key);
    return job ? [job] : [];
  }),
);

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
  | { key: string; createdAtMs: number; kind: "print"; print: Job };

/** One newest-first visual queue across local prints and recovered fleet
 * work. A phase transition changes only the row contents. */
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
  ].sort(compareNewestSubmitted),
);
const active = computed(
  () =>
    running.value.length > 0 ||
    queued.value.length > 0 ||
    partition.value.active.length > 0 ||
    partition.value.attention.length > 0 ||
    props.shared.length > 0,
);
</script>

<template>
  <div v-if="active" class="activity" data-test="activity-strip">
    <div class="activity__head">
      <div class="activity__kicker">Activity</div>
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
            class="activity__row-action"
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

.activity__row-action {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 3px 9px;
  border-radius: var(--radius-control);
  font-family: var(--f-mono);
  font-size: 10.5px;
  cursor: pointer;
}
.activity__row-action:hover {
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
