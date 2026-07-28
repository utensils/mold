<script setup lang="ts">
import { computed, ref, watchEffect } from "vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import {
  mergeActivity,
  sequenceToVM,
  type ActivityAction,
  type ActivityJobVM,
} from "@studio/lib/activity";
import type { ChainJobState } from "@studio/lib/api/chainTypes";
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
 * Create activity strip (Mold Studio). One jobs surface: the running print
 * and queued siblings (generation store) merged with durable sequence jobs
 * from every connected host (chain-jobs store) via the shared @studio
 * activity merge — the old separate chain Jobs list is gone. Hidden when
 * nothing is in flight and no sequences exist.
 */
const emit = defineEmits<{ "edit-sequence": [payload: { hostId: string; jobId: string }] }>();

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

const printVMs = computed<ActivityJobVM[]>(() =>
  generation.pending.map((job) => ({
    kind: "print" as const,
    key: `print:${job.clientId}`,
    hostId: job.hostId ?? "local",
    hostLabel: hostLabel(job.hostId ?? "local"),
    model: job.model,
    prompt: job.prompt,
    phase: job.status === "queued" ? ("queued" as const) : ("running" as const),
    progress: null,
    chain: null,
    actions: ["cancel" as const],
    createdAtMs: job.clientId,
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

/** The merged surface keeps the print chrome; sequence rows render below. */
const sequenceRows = computed(() =>
  mergeActivity(printVMs.value, sequenceVMs.value).filter(
    (vm): vm is ActivityJobVM & { kind: "sequence" } => vm.kind === "sequence",
  ),
);

const show = computed(
  () => !!running.value || queued.value.length > 0 || sequenceRows.value.length > 0,
);

// State color follows the development-temperature rule (design spec §2.1).
const STATE_CLASS: Record<ChainJobState, string> = {
  queued: "text-halide",
  running: "text-safelight",
  interrupted: "text-halide",
  failed: "text-stop",
  completed: "text-ink",
  cancelled: "text-ink-3",
};

const ACTION_LABEL: Record<ActivityAction, string> = {
  watch: "Watch",
  cancel: "Cancel",
  resume: "Resume",
  edit: "Edit",
  delete: "Delete",
  retake: "Retake",
};

const seqPct = (vm: ActivityJobVM) =>
  vm.progress && vm.progress.total > 0
    ? Math.round((vm.progress.step / vm.progress.total) * 100)
    : 0;

const confirmDelete = ref<(ActivityJobVM & { kind: "sequence" }) | null>(null);

function runAction(vm: ActivityJobVM & { kind: "sequence" }, action: ActivityAction) {
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

// ── Strip-level maintenance (mockup Jobs strip) ──────────────────────────────
async function clearInactive() {
  try {
    const { cleared, failed } = await chains.clearInactive();
    generation.prune(0);
    if (failed > 0) toasts.push(`Cleared ${cleared}, ${failed} could not be deleted`, "error");
    else toasts.push(`Cleared ${cleared} job${cleared === 1 ? "" : "s"}`);
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

async function cleanUpDisk() {
  try {
    let swept = 0;
    let pruned = 0;
    for (const hostId of Object.keys(chains.byHost)) {
      const outcome = await chains.gc(hostId);
      swept += outcome.swept_ephemeral_jobs;
      pruned += outcome.pruned_artifact_dirs;
    }
    toasts.push(`Swept ${swept} jobs, pruned ${pruned} dirs`);
  } catch (err) {
    toasts.push(String(err), "error");
  }
}
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
      <template v-if="sequenceRows.length > 0">
        <button
          type="button"
          data-test="activity-clear-inactive"
          class="ms-activity__maint"
          title="Delete all sequence jobs that aren't running or queued and drop finished prints"
          @click="clearInactive"
        >
          Clear inactive
        </button>
        <button
          type="button"
          data-test="activity-cleanup"
          class="ms-activity__maint"
          title="Sweep ephemeral jobs and prune artifact directories on every host"
          @click="cleanUpDisk"
        >
          Clean up disk
        </button>
      </template>
    </div>

    <!-- Sequence rows (merged jobs surface) -->
    <div
      v-for="vm in sequenceRows"
      :key="vm.key"
      class="ms-activity__seq"
      data-test="activity-sequence"
    >
      <span class="ms-activity__state data-mono" :class="STATE_CLASS[vm.state]">{{
        vm.state
      }}</span>
      <span class="ms-activity__seq-model" :title="modelLabel(vm.model)">{{
        modelLabel(vm.model)
      }}</span>
      <span class="ms-activity__seq-meta data-mono">
        {{ vm.stageCount }} clips · {{ Math.min(vm.currentStage + 1, vm.stageCount) }}/{{
          vm.stageCount
        }}
        · {{ vm.hostLabel }}
      </span>
      <div v-if="vm.progress" class="ms-activity__seq-progress" data-test="seq-progress">
        <ProgressBar :value="seqPct(vm)" :height="4" label="Sequence progress" />
      </div>
      <div class="ms-activity__seq-spacer" />
      <span v-if="vm.error" class="ms-activity__seq-error" :title="vm.error">{{ vm.error }}</span>
      <button
        v-for="action in vm.actions"
        :key="action"
        type="button"
        class="ms-activity__seq-btn"
        :class="{ 'ms-activity__seq-btn--danger': action === 'cancel' || action === 'delete' }"
        :data-test="`seq-${action}`"
        @click="runAction(vm, action)"
      >
        {{ ACTION_LABEL[action] }}
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
.ms-activity__seq-progress {
  flex: 0 1 160px;
  min-width: 60px;
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
.ms-activity__seq-btn {
  flex: 0 0 auto;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: 7px;
  padding: 3px 9px;
  font-size: 11px;
  cursor: pointer;
}
.ms-activity__seq-btn:hover {
  color: var(--rebate);
}
.ms-activity__seq-btn--danger:hover {
  color: var(--stop);
  border-color: var(--stop);
}
</style>
