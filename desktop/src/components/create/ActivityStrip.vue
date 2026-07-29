<script setup lang="ts">
import { computed, onBeforeUnmount, reactive, ref, watchEffect } from "vue";
import { useRouter } from "vue-router";
import ProgressBar from "@ui/components/ProgressBar.vue";
import SequenceJobRow from "@ui/components/SequenceJobRow.vue";
import {
  activityDigestLabel,
  mergeActivity,
  partitionActivity,
  sequenceToVM,
  type ActivityAction,
  type ActivityJobVM,
} from "@studio/lib/activity";
import PodCostMeter from "../machines/PodCostMeter.vue";
import ConfirmDialog from "../shell/ConfirmDialog.vue";
import { runPodForHostUrl } from "../../lib/runpod";
import { modelDisplayNameForId } from "../../lib/models";
import { jobProgress, useGenerationStore, type Job } from "../../stores/generation";
import { useChainJobsStore } from "../../stores/chainJobs";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import { useRunPodStore } from "../../stores/runpod";
import { useToastStore } from "../../stores/toasts";

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
const runpod = useRunPodStore();
const toasts = useToastStore();

// ── Prints (unchanged look and behavior) ─────────────────────────────────────
const running = computed<Job | null>(
  () =>
    generation.pending.find(
      (j) => j.status === "denoising" || j.status === "finishing" || j.status === "loading",
    ) ?? null,
);
const queued = computed<Job[]>(() => generation.pending.filter((j) => j.status === "queued"));
const runningPct = computed(() =>
  running.value ? Math.round(jobProgress(running.value) * 100) : 0,
);
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
  void generation.cancel(job.clientId).then(() => toasts.push("Cancelled"));
}

// ── Sequences via the shared activity merge ──────────────────────────────────
const hostLabel = (hostId: string) => hosts.all.find((h) => h.id === hostId)?.label ?? hostId;
const modelLabel = (name: string) => modelDisplayNameForId(name, hostModels.unionInstalled);

/** Every print, not just the pending ones: a failed print earns an attention
 *  row, so it has to survive the merge. `createdAtMs` is a real wall clock —
 *  the old `job.clientId` counter sorted every print below every sequence. */
const printVMs = computed<ActivityJobVM[]>(() =>
  generation.jobs.map((job) => ({
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
  })),
);

const sequenceVMs = computed<ActivityJobVM[]>(() =>
  chains.allJobs.map(({ hostId, job }) => {
    const watched = chains.watching?.hostId === hostId && chains.watching.jobId === job.id;
    const active = chains.live.activeStage;
    const progress = watched && active !== null ? (chains.live.progress[active] ?? null) : null;
    return sequenceToVM(job, { hostId, hostLabel: hostLabel(hostId) }, progress);
  }),
);

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
        <div class="ms-activity__running">
          <div class="ms-activity__line">
            <span class="ms-activity__prompt">{{ running.prompt }}</span>
            <PodCostMeter
              v-if="runningPod"
              data-test="activity-pod-cost"
              class="ms-activity__cost"
              :cost-per-hr="runningPod.costPerHr"
              :uptime-seconds="runningPod.uptimeSeconds"
            />
            <span class="ms-activity__pct data-mono">{{ runningPct }}%</span>
          </div>
          <ProgressBar :value="runningPct" :height="4" label="Print progress" />
        </div>
      </template>
      <div v-else class="ms-activity__idle-spacer" />
      <div
        v-for="job in queued"
        :key="job.clientId"
        class="ms-activity__pill"
        data-test="activity-queued"
      >
        <span class="ms-activity__pill-text">Queued · {{ job.prompt }}</span>
        <button
          type="button"
          class="ms-activity__cancel"
          :aria-label="`Cancel queued print: ${job.prompt}`"
          @click="cancel(job)"
        >
          ✕
        </button>
      </div>
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

    <!-- In-flight sequence rows (merged jobs surface) -->
    <SequenceJobRow
      v-for="vm in sequenceRows"
      :key="vm.key"
      data-test="activity-sequence"
      :vm="vm"
      :model-label="modelLabel(vm.model)"
      @action="runAction"
    />

    <!-- Settled but still wanting a decision: capped, expiring, dismissible. -->
    <SequenceJobRow
      v-for="vm in attentionSequences"
      :key="vm.key"
      data-test="activity-sequence"
      :vm="vm"
      :model-label="modelLabel(vm.model)"
      dismissible
      @action="runAction"
      @dismiss="dismiss"
    />

    <div
      v-for="vm in attentionPrints"
      :key="vm.key"
      class="ms-activity__seq"
      data-test="activity-print-attention"
      role="alert"
    >
      <span class="ms-activity__state data-mono text-stop">failed</span>
      <span class="ms-activity__seq-model" :title="vm.prompt">{{ vm.prompt }}</span>
      <div class="ms-activity__seq-spacer" />
      <span v-if="vm.error" class="ms-activity__seq-error" :title="vm.error">{{ vm.error }}</span>
      <button
        type="button"
        class="ms-activity__cancel"
        data-test="print-dismiss"
        title="Hide this failure. Nothing is deleted."
        :aria-label="`Dismiss failed print: ${vm.prompt}`"
        @click="dismiss(vm)"
      >
        ✕
      </button>
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
}
.ms-activity__row {
  display: flex;
  align-items: center;
  gap: 14px;
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
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  gap: 6px;
  background: var(--bath);
  border: 1px solid var(--edge);
  border-radius: 20px;
  padding: 5px 5px 5px 11px;
}
.ms-activity__pill-text {
  font-size: 11px;
  color: var(--ink-2);
  max-width: 150px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
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
