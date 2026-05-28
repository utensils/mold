<script setup lang="ts">
import { computed } from "vue";
import type { Job } from "../composables/useGenerateStream";
import type { GenerateRequestWire, QueueEntry } from "../types";
import RunningJobCard from "./RunningJobCard.vue";

type GpuLane = { ordinal: number; state?: string };
type LaneKey = "auto" | number;
type DisplayItem = {
  job: Job;
  queueEntry: QueueEntry | null;
  lane: LaneKey;
  sortState: "running" | "queued" | "settled";
  position: number;
};

const props = withDefaults(
  defineProps<{
    jobs: Job[];
    queueEntries?: QueueEntry[];
    gpus?: GpuLane[];
  }>(),
  {
    queueEntries: () => [],
    gpus: () => [],
  },
);
const emit = defineEmits<{
  (e: "cancel", id: string): void;
  (e: "open", job: Job): void;
  (e: "dismiss", id: string): void;
  (e: "clear-finished"): void;
  (e: "lane-change", id: string, targetGpu: number | null): void;
}>();

const hasFinished = computed(() =>
  props.jobs.some((j) => j.state !== "running"),
);

function queueLane(entry: QueueEntry | null, job: Job): LaneKey {
  if (entry?.state === "running")
    return entry.gpu ?? job.progress.gpu ?? "auto";
  if (entry?.state === "queued") {
    return entry.target_gpu ?? entry.preferred_gpu ?? "auto";
  }
  return job.progress.gpu ?? "auto";
}

function queueSortState(
  entry: QueueEntry | null,
  job: Job,
): DisplayItem["sortState"] {
  if (entry?.state === "running") return "running";
  if (entry?.state === "queued") return "queued";
  if (job.state === "running" && job.workStarted) return "running";
  if (job.state === "running") return "queued";
  return "settled";
}

function displayJobForLocal(job: Job, entry: QueueEntry | null): Job {
  if (!entry) return job;
  const progress = { ...job.progress };
  if (entry.state === "running") {
    progress.gpu = entry.gpu ?? progress.gpu;
    progress.queuePosition = null;
  } else {
    progress.queuePosition = entry.position;
    progress.stage = `Queued (position ${entry.position})`;
    progress.step = null;
    progress.totalSteps = null;
    progress.gpu = null;
  }
  return {
    ...job,
    progress,
    workStarted: entry.state === "running" ? job.workStarted : false,
  };
}

function displayJobForServerEntry(entry: QueueEntry): Job {
  const request = {
    prompt: "",
    model: entry.model,
    width: 0,
    height: 0,
    steps: 0,
    guidance: 0,
  } as GenerateRequestWire;
  return {
    id: `server:${entry.id}`,
    request,
    startedAt: entry.started_at_unix_ms,
    controller: new AbortController(),
    progress: {
      stage:
        entry.state === "queued"
          ? `Queued (position ${entry.position})`
          : "Running",
      step: null,
      totalSteps: null,
      weightBytesLoaded: null,
      weightBytesTotal: null,
      queuePosition: entry.state === "queued" ? entry.position : null,
      gpu: entry.state === "running" ? (entry.gpu ?? null) : null,
      elapsedMs: null,
    },
    result: null,
    error: null,
    state: "running",
    chain: null,
    lastProgressAt: Date.now(),
    workStarted: entry.state === "running",
    serverId: entry.id,
  };
}

const displayItems = computed<DisplayItem[]>(() => {
  const queueById = new Map(props.queueEntries.map((e) => [e.id, e]));
  const seen = new Set<string>();
  const items: DisplayItem[] = [];

  for (const job of props.jobs) {
    const entry = job.serverId ? (queueById.get(job.serverId) ?? null) : null;
    if (entry) seen.add(entry.id);
    const merged = displayJobForLocal(job, entry);
    items.push({
      job: merged,
      queueEntry: entry,
      lane: queueLane(entry, merged),
      sortState: queueSortState(entry, merged),
      position: entry?.position ?? Number.MAX_SAFE_INTEGER,
    });
  }

  for (const entry of props.queueEntries) {
    if (seen.has(entry.id)) continue;
    const job = displayJobForServerEntry(entry);
    items.push({
      job,
      queueEntry: entry,
      lane: queueLane(entry, job),
      sortState: entry.state,
      position: entry.position,
    });
  }

  const rank = { running: 0, queued: 1, settled: 2 };
  return items.sort(
    (a, b) => rank[a.sortState] - rank[b.sortState] || a.position - b.position,
  );
});

const stateRank = { running: 0, queued: 1, settled: 2 };

const lanes = computed(() => {
  const keys: LaneKey[] = [];
  if (displayItems.value.some((i) => i.lane === "auto")) keys.push("auto");
  for (const gpu of props.gpus) keys.push(gpu.ordinal);
  for (const item of displayItems.value) {
    if (item.lane !== "auto" && !keys.includes(item.lane)) keys.push(item.lane);
  }
  return keys
    .map((key) => ({
      key,
      label: key === "auto" ? "Auto" : `GPU ${key}`,
      items: displayItems.value.filter((i) => i.lane === key),
    }))
    .filter((lane) => lane.items.length > 0 || lane.key !== "auto")
    .sort((a, b) => {
      const aRank = Math.min(
        ...a.items.map((i) => stateRank[i.sortState]),
        Number.MAX_SAFE_INTEGER,
      );
      const bRank = Math.min(
        ...b.items.map((i) => stateRank[i.sortState]),
        Number.MAX_SAFE_INTEGER,
      );
      if (aRank !== bRank) return aRank - bRank;
      if (a.key === "auto") return -1;
      if (b.key === "auto") return 1;
      return Number(a.key) - Number(b.key);
    });
});
</script>

<template>
  <div v-if="displayItems.length" class="mt-4 flex flex-col gap-2">
    <div
      v-for="lane in lanes"
      :key="lane.key"
      class="flex flex-col gap-2"
      :data-test="
        lane.key === 'auto' ? 'queue-lane-auto' : `queue-lane-gpu-${lane.key}`
      "
    >
      <div class="text-xs font-medium uppercase tracking-wide text-slate-500">
        {{ lane.label }}
      </div>
      <div class="flex gap-3 overflow-x-auto pb-2">
        <RunningJobCard
          v-for="item in lane.items"
          :key="item.job.id"
          :job="item.job"
          :queue-entry="item.queueEntry"
          :gpus="gpus"
          @cancel="(id: string) => emit('cancel', id)"
          @open="(j: Job) => emit('open', j)"
          @dismiss="(id: string) => emit('dismiss', id)"
          @lane-change="
            (id: string, targetGpu: number | null) =>
              emit('lane-change', id, targetGpu)
          "
        />
      </div>
    </div>
    <div v-if="hasFinished" class="flex justify-end">
      <button
        type="button"
        class="text-xs text-slate-400 hover:text-slate-200"
        data-test="clear-finished"
        @click="emit('clear-finished')"
      >
        Clear finished
      </button>
    </div>
  </div>
</template>
