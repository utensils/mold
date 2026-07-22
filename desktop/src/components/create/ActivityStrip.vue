<script setup lang="ts">
import { computed, watchEffect } from "vue";
import ProgressBar from "@ui/components/ProgressBar.vue";
import PodCostMeter from "../machines/PodCostMeter.vue";
import { runPodForHostUrl } from "../../lib/runpod";
import { jobProgress, useGenerationStore, type Job } from "../../stores/generation";
import { useHostsStore } from "../../stores/hosts";
import { useRunPodStore } from "../../stores/runpod";
import { useToastStore } from "../../stores/toasts";

/**
 * Create activity strip (Mold Studio). Mirrors the running print and the
 * queued siblings using the generation store the canvas already tracks, so the
 * strip and the canvas never disagree. Hidden when nothing is in flight.
 */
const generation = useGenerationStore();
const hosts = useHostsStore();
const runpod = useRunPodStore();
const toasts = useToastStore();

const running = computed<Job | null>(
  () =>
    generation.pending.find(
      (j) => j.status === "denoising" || j.status === "finishing" || j.status === "loading",
    ) ?? null,
);
const queued = computed<Job[]>(() => generation.pending.filter((j) => j.status === "queued"));
const show = computed(() => !!running.value || queued.value.length > 0);
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
</script>

<template>
  <div v-if="show" data-test="activity-strip" class="ms-activity">
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
  </div>
</template>

<style scoped>
.ms-activity {
  border-top: 1px solid var(--edge);
  background: var(--bench);
  padding: 9px 22px 10px;
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
</style>
