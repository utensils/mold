<script setup lang="ts">
import { computed, onBeforeUnmount, reactive, ref, watch, watchEffect } from "vue";
import { useRouter } from "vue-router";
import ProgressBar from "@ui/components/ProgressBar.vue";
import LiveActivityList from "@ui/components/LiveActivityList.vue";
import SequenceJobRow from "@ui/components/SequenceJobRow.vue";
import {
  activityDigestLabel,
  mergeActivity,
  partitionActivity,
  queueStatusLabel,
  sequenceToVM,
  withLiveQueueStatus,
  type ActivityAction,
  type ActivityJobVM,
  type PrintActivityVM,
} from "@studio/lib/activity";
import { buildQueueStatusIndex } from "@studio/lib/queuePosition";
import PodCostMeter from "../machines/PodCostMeter.vue";
import ConfirmDialog from "../shell/ConfirmDialog.vue";
import { runPodForHostUrl } from "../../lib/runpod";
import { modelDisplayNameForId } from "../../lib/models";
import { jobProgress, useGenerationStore, type Job } from "../../stores/generation";
import { useChainJobsStore } from "../../stores/chainJobs";
import { useJobsStore } from "../../stores/jobs";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import { useRunPodStore } from "../../stores/runpod";
import { useToastStore } from "../../stores/toasts";
import { useComposerStore } from "../../stores/composer";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { useLiveActivityStore } from "../../stores/liveActivity";
import { useOpenLiveWork } from "../../composables/useOpenLiveWork";

/**
 * Create activity strip (Mold Studio) — present tense only.
 *
 * It shows work that is happening (running print + queued siblings + queued /
 * running sequences), plus a capped, expiring set of settled-but-wrong rows
 * that still want a decision, plus one digest chip counting the settled
 * sequences it is deliberately not listing. Settled work resolves to the
 * Library: the print in the grid, the durable job in Library ▸ History ▸
 * Sequences, which is also where Clear inactive / Clean up disk now live.
 */
const emit = defineEmits<{ "edit-sequence": [payload: { hostId: string; jobId: string }] }>();

const router = useRouter();
const generation = useGenerationStore();
const hosts = useHostsStore();
const hostModels = useHostModelsStore();
const chains = useChainJobsStore();
const jobsStore = useJobsStore();
const runpod = useRunPodStore();
const toasts = useToastStore();
const composer = useComposerStore();
const draft = useSequenceDraftStore();
const liveActivity = useLiveActivityStore();
const openLiveWork = useOpenLiveWork();

function selectPrint(job: Job) {
  generation.select(job.clientId);
  draft.stopEditing();
  draft.output = "single";
  const request = job.request;
  if (request) composer.set({ request });
}

function selectPrintVm(vm: ActivityJobVM) {
  const job = generation.jobs.find((candidate) => `print:${candidate.clientId}` === vm.key);
  if (job) selectPrint(job);
}

function selectSequence(vm: ActivityJobVM & { kind: "sequence" }) {
  composer.setSequence({ kind: "inspect", hostId: vm.hostId, jobId: vm.jobId });
}

// ── Prints (unchanged look and behavior) ─────────────────────────────────────
const running = computed<Job | null>(
  () =>
    generation.pending.find(
      (j) => j.status === "denoising" || j.status === "finishing" || j.status === "loading",
    ) ?? null,
);
const queued = computed<Job[]>(() => generation.pending.filter((j) => j.status === "queued"));

/**
 * Live dispatch order for our own queued prints. The SSE `Queued { position }`
 * frame is a one-shot — it never fires again as the queue drains — so the pill
 * reads the host's `/api/queue` listing through the same store the Machines
 * queue column uses. The poll is retained only while something is actually
 * waiting, so an idle Create window costs nothing.
 */
const queueStatus = computed(() =>
  buildQueueStatusIndex(
    Object.values(jobsStore.queues).map((snapshot) => ({
      hostId: snapshot.hostId,
      entries: snapshot.entries,
      plan: snapshot.plan,
    })),
  ),
);

let retainedQueuePoll = false;
function retainQueuePoll(want: boolean) {
  if (want === retainedQueuePoll) return;
  retainedQueuePoll = want;
  if (want) jobsStore.startPolling();
  else jobsStore.stopPolling();
}
watch(() => queued.value.length > 0, retainQueuePoll, { immediate: true });
onBeforeUnmount(() => retainQueuePoll(false));

const MAX_QUEUED_PILLS = 4;
const visibleQueued = computed(() => queued.value.slice(0, MAX_QUEUED_PILLS));
const hiddenQueuedCount = computed(() =>
  Math.max(0, queued.value.length - visibleQueued.value.length),
);
const runningPct = computed(() =>
  running.value ? Math.round(jobProgress(running.value) * 100) : 0,
);
const runningFinalizing = computed(() => running.value?.status === "finishing");
const runningHost = computed(() =>
  running.value?.hostId
    ? (hosts.all.find((host) => host.id === running.value?.hostId) ?? null)
    : null,
);
const runningPod = computed(() => runPodForHostUrl(runpod.runningPods, runningHost.value?.baseUrl));

watchEffect(() => {
  const url = runningHost.value?.baseUrl ?? "";
  if (url.includes(".proxy.runpod.net") && !runpod.loaded && !runpod.loading) {
    void runpod.load();
  }
});

function cancel(job: Job) {
  if (job.cancelling) return;
  void generation
    .cancel(job.clientId)
    .then((cancelled) => {
      if (cancelled) toasts.push("Cancelled");
    })
    .catch((error) => toasts.push(error instanceof Error ? error.message : String(error), "error"));
}

function retry(job: Job) {
  if (!job.retryable || job.retrying) return;
  void generation
    .retryHeld(job.clientId)
    .then(() => toasts.push(`Retry queued on ${job.hostLabel ?? "this machine"}.`))
    .catch((error) => toasts.push(error instanceof Error ? error.message : String(error), "error"));
}

// ── Sequences via the shared activity merge ──────────────────────────────────
const hostLabel = (hostId: string) => hosts.all.find((h) => h.id === hostId)?.label ?? hostId;
const modelLabel = (name: string) => modelDisplayNameForId(name, hostModels.unionInstalled);

/** Every print, not just the pending ones: a failed print earns an attention
 *  row, so it has to survive the merge. `createdAtMs` is a real wall clock —
 *  the old `job.clientId` counter sorted every print below every sequence. */
const printVMs = computed<ActivityJobVM[]>(() =>
  generation.jobs.map((job) => {
    const vm: PrintActivityVM = {
      kind: "print" as const,
      key: `print:${job.clientId}`,
      hostId: job.hostId ?? "local",
      hostLabel: hostLabel(job.hostId ?? "local"),
      model: job.model,
      prompt: job.prompt,
      phase:
        job.status === "queued"
          ? ("queued" as const)
          : job.status === "error"
            ? ("failed" as const)
            : job.status === "complete"
              ? ("done" as const)
              : ("running" as const),
      progress: null,
      chain: null,
      actions: ["cancel" as const],
      createdAtMs: job.submittedAtUnixMs,
      settledAtMs: job.settledAtMs,
      error: job.error,
    };
    return withLiveQueueStatus(vm, queueStatus.value, job.id);
  }),
);

/** "Next up" / "#2 in line" / "Waiting for memory" per queued pill, resolved in
 *  the shared studio layer so this pill and web's say the same thing. A host
 *  too old to list the job degrades to the plain "Queued". */
const queueLabelByKey = computed(() => {
  const labels = new Map<string, string>();
  for (const vm of printVMs.value) {
    const label = queueStatusLabel(vm);
    if (label) labels.set(vm.key, label);
  }
  return labels;
});
function queuedLabel(job: Job): string {
  return queueLabelByKey.value.get(`print:${job.clientId}`) ?? "Queued";
}

/** The presentation's own label for a settled print whose outcome is not
 *  knowable here; `null` for a real failure. */
const advisoryByKey = computed(
  () =>
    new Map(
      generation.jobs
        .filter((job) => job.outcomeUnknown)
        .map((job) => [`print:${job.clientId}`, job.stage ?? "Outcome unknown"]),
    ),
);
function advisoryLabel(vm: ActivityJobVM): string | null {
  return advisoryByKey.value.get(vm.key) ?? null;
}

const sequenceVMs = computed<ActivityJobVM[]>(() =>
  chains.allJobs.map(({ hostId, job }) => {
    const watched = chains.watching?.hostId === hostId && chains.watching.jobId === job.id;
    const active = chains.live.activeStage;
    const progress = watched && active !== null ? (chains.live.progress[active] ?? null) : null;
    return sequenceToVM(job, { hostId, hostLabel: hostLabel(hostId) }, progress);
  }),
);

const sharedRows = computed(() => {
  const primaryId = hosts.primaryHost?.id ?? "local";
  const local = new Set(
    generation.jobs.flatMap((job) =>
      job.id ? [`${job.hostId ?? primaryId}:generation:${job.id}`] : [],
    ),
  );
  for (const { hostId, job } of chains.allJobs) local.add(`${hostId}:sequence:${job.id}`);
  return liveActivity.rows.filter((row) => !local.has(row.key));
});

/** Session-only dismissals keyed by VM key. Deliberately NOT persisted: the
 *  5-minute age rule reads server timestamps and already survives a restart,
 *  and a second retention mechanism could disagree with the first. reactive()
 *  because rows are dismissed from click handlers. */
const dismissed = reactive(new Set<string>());

/** Re-partition on a timer so an attention row actually expires while the
 *  window stays open, instead of waiting for the next store mutation. */
const nowMs = ref(Date.now());
const clock = setInterval(() => (nowMs.value = Date.now()), 30_000);
onBeforeUnmount(() => clearInterval(clock));

const partition = computed(() =>
  partitionActivity(mergeActivity(printVMs.value, sequenceVMs.value), {
    nowMs: nowMs.value,
    dismissed,
  }),
);

/** Active sequences render as rows; active prints keep the strip's own
 *  running/queued chrome above. */
const sequenceRows = computed(() =>
  partition.value.active.filter(
    (vm): vm is ActivityJobVM & { kind: "sequence" } => vm.kind === "sequence",
  ),
);
const attentionSequences = computed(() =>
  partition.value.attention.filter(
    (vm): vm is ActivityJobVM & { kind: "sequence" } => vm.kind === "sequence",
  ),
);
const attentionPrints = computed(() =>
  partition.value.attention.filter(
    (vm): vm is ActivityJobVM & { kind: "print" } => vm.kind === "print",
  ),
);
const digest = computed(() => activityDigestLabel(partition.value));

function dismiss(vm: ActivityJobVM) {
  dismissed.add(vm.key);
}

function openHistory() {
  void router.push({ path: "/library", query: { panel: "history", tab: "sequences" } });
}

const show = computed(
  () =>
    !!running.value ||
    queued.value.length > 0 ||
    sequenceRows.value.length > 0 ||
    sharedRows.value.length > 0 ||
    partition.value.attention.length > 0 ||
    digest.value !== null,
);

const confirmDelete = ref<(ActivityJobVM & { kind: "sequence" }) | null>(null);

function runAction(action: ActivityAction, vm: ActivityJobVM & { kind: "sequence" }) {
  switch (action) {
    case "watch":
      chains.watch(vm.hostId, vm.jobId);
      return;
    case "cancel":
      void chains.cancel(vm.hostId, vm.jobId).catch((err) => toasts.push(String(err), "error"));
      return;
    case "resume":
      void chains.resume(vm.hostId, vm.jobId).catch((err) => toasts.push(String(err), "error"));
      return;
    case "edit":
      emit("edit-sequence", { hostId: vm.hostId, jobId: vm.jobId });
      return;
    case "delete":
      confirmDelete.value = vm;
      return;
    default:
      return;
  }
}

function deleteConfirmed() {
  const vm = confirmDelete.value;
  confirmDelete.value = null;
  if (!vm) return;
  void chains.remove(vm.hostId, vm.jobId).catch((err) => toasts.push(String(err), "error"));
}

// `Clear inactive` and `Clean up disk` deliberately left this strip: they are
// destructive, host-scoped, and rarely used, and their presence here is what
// made the composer read as a control panel. They live in the History ▸
// Sequences footer now.
</script>

<template>
  <div v-if="show" data-test="activity-strip" class="ms-activity">
    <div class="ms-activity__row">
      <span class="ms-activity__kicker">Activity</span>
      <template v-if="running">
        <span class="ms-activity__thumb ms-shimmer" aria-hidden="true" />
        <button
          type="button"
          class="ms-activity__running text-left"
          data-test="activity-running-select"
          @click="selectPrint(running)"
        >
          <div class="ms-activity__line">
            <span class="ms-activity__prompt">{{ running.prompt }}</span>
            <PodCostMeter
              v-if="runningPod"
              data-test="activity-pod-cost"
              class="ms-activity__cost"
              :cost-per-hr="runningPod.costPerHr"
              :uptime-seconds="runningPod.uptimeSeconds"
            />
            <span class="ms-activity__pct data-mono">
              {{ runningFinalizing ? "Finalizing" : `${runningPct}%` }}
            </span>
          </div>
          <ProgressBar
            :value="runningPct"
            :height="4"
            :label="runningFinalizing ? 'Finalizing print' : 'Print progress'"
          />
        </button>
        <button
          type="button"
          class="ms-activity__cancel"
          data-test="activity-running-cancel"
          :disabled="running.cancelling"
          :aria-label="
            running.cancelling ? `Cancelling ${running.prompt}` : `Cancel ${running.prompt}`
          "
          @click="cancel(running)"
        >
          {{ running.cancelling ? "…" : "✕" }}
        </button>
      </template>
      <div v-else class="ms-activity__idle-spacer" />
      <div
        v-for="job in visibleQueued"
        :key="job.clientId"
        class="ms-activity__pill"
        data-test="activity-queued"
        role="button"
        tabindex="0"
        :title="`${queuedLabel(job)} · ${job.prompt}`"
        @click="selectPrint(job)"
        @keydown.enter.prevent="selectPrint(job)"
        @keydown.space.prevent="selectPrint(job)"
      >
        <span class="ms-activity__pill-text">
          <span class="ms-activity__pill-queue" data-test="activity-queued-position">{{
            queuedLabel(job)
          }}</span>
          · {{ job.prompt }}
          <span v-if="job.holdError" class="ms-activity__seq-error"> · {{ job.holdError }}</span>
        </span>
        <button
          v-if="job.retryable"
          type="button"
          class="ms-activity__seq-action"
          data-test="activity-held-retry"
          :disabled="job.retrying"
          @click.stop="retry(job)"
        >
          {{ job.retrying ? "Retrying…" : "Retry" }}
        </button>
        <button
          type="button"
          class="ms-activity__cancel"
          :aria-label="
            job.cancelling ? `Cancelling ${job.prompt}` : `Cancel queued print: ${job.prompt}`
          "
          :disabled="job.cancelling"
          @click.stop="cancel(job)"
        >
          {{ job.cancelling ? "…" : "✕" }}
        </button>
      </div>
      <span
        v-if="hiddenQueuedCount > 0"
        class="ms-activity__overflow data-mono"
        data-test="activity-queued-overflow"
      >
        +{{ hiddenQueuedCount }} queued
      </span>
      <!-- Everything settled the strip is not listing, in one mono count. -->
      <button
        v-if="digest"
        type="button"
        data-test="activity-digest"
        class="ms-activity__maint"
        title="Show settled sequences in History"
        @click="openHistory"
      >
        {{ digest }}
      </button>
    </div>

    <div data-test="activity-list-scroll" class="ms-activity__list">
      <LiveActivityList :rows="sharedRows" interactive @select="openLiveWork" />

      <!-- In-flight sequence rows (merged jobs surface) -->
      <SequenceJobRow
        v-for="vm in sequenceRows"
        :key="vm.key"
        data-test="activity-sequence"
        :vm="vm"
        :model-label="modelLabel(vm.model)"
        :show-error="false"
        @action="runAction"
        @select="selectSequence"
      />

      <!-- Settled but still wanting a decision: capped, expiring, dismissible. -->
      <SequenceJobRow
        v-for="vm in attentionSequences"
        :key="vm.key"
        data-test="activity-sequence"
        :vm="vm"
        :model-label="modelLabel(vm.model)"
        :show-error="false"
        dismissible
        @action="runAction"
        @select="selectSequence"
        @dismiss="dismiss"
      />

      <div
        v-for="vm in attentionPrints"
        :key="vm.key"
        class="ms-activity__seq"
        data-test="activity-print-attention"
        role="alert"
        tabindex="0"
        @click="selectPrintVm(vm)"
        @keydown.enter.prevent="selectPrintVm(vm)"
        @keydown.space.prevent="selectPrintVm(vm)"
      >
        <span
          class="ms-activity__state data-mono"
          :class="advisoryLabel(vm) ? 'text-ink-2' : 'text-stop'"
          >{{ advisoryLabel(vm) ?? "failed" }}</span
        >
        <span class="ms-activity__seq-model" :title="vm.prompt">{{ vm.prompt }}</span>
        <div class="ms-activity__seq-spacer" />
        <span class="ms-activity__seq-error">Open Create for details</span>
        <button
          type="button"
          class="ms-activity__cancel"
          data-test="print-dismiss"
          :title="
            advisoryLabel(vm)
              ? 'Hide this row. Nothing is deleted.'
              : 'Hide this failure. Nothing is deleted.'
          "
          :aria-label="`Dismiss ${advisoryLabel(vm) ? 'print' : 'failed print'}: ${vm.prompt}`"
          @click.stop="dismiss(vm)"
        >
          ✕
        </button>
      </div>
    </div>

    <ConfirmDialog
      :open="confirmDelete !== null"
      title="Delete this sequence?"
      :message="`Removes the job and its cached clips${confirmDelete ? ` from ${confirmDelete.hostLabel}` : ''}. The finished video in the Library is kept.`"
      confirm-label="Delete"
      danger
      @confirm="deleteConfirmed"
      @cancel="confirmDelete = null"
    />
  </div>
</template>

<style scoped>
.ms-activity {
  border-top: 1px solid var(--edge);
  background: var(--bench);
  padding: 9px 22px 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
  min-height: 0;
  max-height: min(42cqh, 220px);
  overflow: hidden;
  flex-shrink: 1;
}
.ms-activity__list {
  display: flex;
  min-height: 0;
  flex-direction: column;
  gap: 8px;
  overflow-x: hidden;
  overflow-y: auto;
  overscroll-behavior: contain;
  scrollbar-gutter: stable;
}
.ms-activity__row {
  display: flex;
  align-items: center;
  gap: 14px;
  min-width: 0;
  overflow: hidden;
}
.ms-activity__kicker {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
  flex: 0 0 auto;
}
.ms-activity__thumb {
  width: 26px;
  height: 26px;
  flex: 0 0 26px;
  border-radius: 6px;
  background: var(--print);
}
.ms-activity__running {
  flex: 1;
  min-width: 0;
}
.ms-activity__idle-spacer {
  flex: 1;
}
.ms-activity__line {
  display: flex;
  gap: 8px;
  align-items: baseline;
}
.ms-activity__prompt {
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
}
.ms-activity__pct {
  font-size: 9.5px;
  color: var(--safelight);
  flex: 0 0 auto;
}
.ms-activity__cost {
  flex: 0 0 auto;
  font-size: 9.5px;
}
.ms-activity__pill {
  flex: 1 1 120px;
  min-width: 0;
  max-width: 166px;
  display: flex;
  align-items: center;
  gap: 6px;
  background: var(--bath);
  border: 1px solid var(--edge);
  border-radius: 20px;
  padding: 5px 5px 5px 11px;
}
.ms-activity__overflow {
  flex: 0 0 auto;
  color: var(--ink-3);
  font-size: 9.5px;
  white-space: nowrap;
}
.ms-activity__pill-text {
  font-size: 11px;
  color: var(--ink-2);
  max-width: 150px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ms-activity__pill-queue {
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--safelight);
}
.ms-activity__cancel {
  width: 18px;
  height: 18px;
  border: 0;
  background: transparent;
  color: var(--ink-3);
  border-radius: 50%;
  font-size: 11px;
  cursor: pointer;
}
.ms-activity__maint {
  flex: 0 0 auto;
  border: 0;
  background: transparent;
  font-family: var(--f-mono);
  font-size: 9.5px;
  letter-spacing: 0.04em;
  color: var(--ink-3);
  cursor: pointer;
}
.ms-activity__maint:hover {
  color: var(--rebate);
}
.ms-activity__seq {
  display: flex;
  align-items: center;
  gap: 10px;
}
.ms-activity__state {
  font-size: 9.5px;
  text-transform: uppercase;
  flex: 0 0 auto;
}
.ms-activity__seq-model {
  font-size: 12px;
  color: var(--rebate);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 220px;
}
.ms-activity__seq-meta {
  font-size: 9.5px;
  color: var(--ink-3);
  flex: 0 0 auto;
}
.ms-activity__seq-spacer {
  flex: 1;
}
.ms-activity__seq-error {
  font-size: 10.5px;
  color: var(--stop);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 320px;
}
</style>
