<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { useRouter } from "vue-router";
import DevelopCanvas from "../lib/develop/DevelopCanvas.vue";
import { useGenerationStore, jobPhase, jobProgress, type Job } from "../stores/generation";
import { useHostsStore, type HostView } from "../stores/hosts";
import { enrichQueueEntries, useJobsStore, type EnrichedQueueEntry } from "../stores/jobs";
import { useComposerStore } from "../stores/composer";
import { useToastStore } from "../stores/toasts";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import {
  computeQueueLanes,
  laneForEntry,
  resolveDropAction,
  type DraggedQueueRow,
  type LaneKey,
  type QueueLane,
} from "../lib/queueLanes";

const router = useRouter();
const generation = useGenerationStore();
const hosts = useHostsStore();
const jobs = useJobsStore();
const composer = useComposerStore();
const toasts = useToastStore();

onMounted(() => jobs.startPolling());
onUnmounted(() => jobs.stopPolling());

const primaryId = computed(() => hosts.primaryHost?.id ?? null);

/**
 * Per-host lanes, computed once per reactive update and cached by host id so
 * row/lane rendering never re-enriches or re-groups the queue for every row.
 */
const laneCache = computed(() => {
  const map = new Map<string, QueueLane<EnrichedQueueEntry>[]>();
  for (const host of hosts.all) {
    const snapshot = jobs.queues[host.id];
    const entries = snapshot
      ? enrichQueueEntries(snapshot.entries, host.id, generation.jobs, primaryId.value)
      : [];
    map.set(host.id, computeQueueLanes(entries, snapshot?.gpuOrdinals ?? []));
  }
  return map;
});

function hostLanes(host: HostView): QueueLane<EnrichedQueueEntry>[] {
  return laneCache.value.get(host.id) ?? [];
}

/** Whether a host has any queued/running rows across all its lanes. */
function hostHasEntries(host: HostView): boolean {
  return hostLanes(host).some((lane) => lane.entries.length > 0);
}

/** This app's job behind a queue row, for thumbnails and live progress. */
function ownJob(entry: EnrichedQueueEntry): Job | null {
  if (entry.clientId === null) return null;
  return generation.jobs.find((j) => j.clientId === entry.clientId) ?? null;
}

function entryCode(entry: EnrichedQueueEntry): string {
  if (entry.state === "running") {
    const job = ownJob(entry);
    if (job && job.total > 0 && job.status === "denoising") return `${job.step}/${job.total}`;
    return entry.gpu !== undefined ? `RUNNING · GPU ${entry.gpu}` : "RUNNING";
  }
  return entry.position > 0 ? `QUEUED #${entry.position}` : "QUEUED";
}

// ── Per-GPU lanes (multi-GPU hosts only) ─────────────────────────────────
const contextMenu = useContextMenuStore();
const dragged = ref<DraggedQueueRow | null>(null);

/** Numeric lane keys of a host — empty means the flat single-queue layout. */
function laneOrdinals(host: HostView): number[] {
  return hostLanes(host)
    .map((l) => l.key)
    .filter((k): k is number => k !== "queue");
}

function canDragEntry(host: HostView, entry: EnrichedQueueEntry): boolean {
  return entry.state === "queued" && laneOrdinals(host).length > 0;
}

function onDragStart(host: HostView, entry: EnrichedQueueEntry, event: DragEvent) {
  if (!canDragEntry(host, entry)) {
    dragged.value = null;
    return;
  }
  dragged.value = { hostId: host.id, entryId: entry.id, lane: laneForEntry(entry) };
  event.dataTransfer?.setData("text/plain", entry.id);
  if (event.dataTransfer) event.dataTransfer.effectAllowed = "move";
}

function onLaneDrop(host: HostView, laneKey: LaneKey) {
  const action = resolveDropAction(dragged.value, host.id, laneKey);
  dragged.value = null;
  if (action.kind === "reject-cross-host") {
    toasts.push("Jobs can't move between hosts.", "error");
    return;
  }
  if (action.kind === "reassign") {
    void jobs.reassignGpu(action.hostId, action.entryId, action.targetGpu);
  }
}

/** Accessible non-drag fallback: right-click → Move to GPU N. */
function openEntryMenu(host: HostView, entry: EnrichedQueueEntry, event: MouseEvent) {
  const ordinals = laneOrdinals(host);
  if (entry.state !== "queued" || ordinals.length === 0) return;
  const current = laneForEntry(entry);
  const items: MenuEntry[] = ordinals.map((ordinal) => ({
    label: `Move to GPU ${ordinal}`,
    disabled: ordinal === current,
    action: () => void jobs.reassignGpu(host.id, entry.id, ordinal),
  }));
  contextMenu.open(event, items);
}

/** Highlight a lane while a same-host drag could land on it. */
function laneDroppable(host: HostView, laneKey: LaneKey): boolean {
  return (
    dragged.value !== null &&
    dragged.value.hostId === host.id &&
    laneKey !== "queue" &&
    laneKey !== dragged.value.lane
  );
}

async function cancelEntry(host: HostView, entry: EnrichedQueueEntry) {
  try {
    if (entry.clientId !== null) await generation.cancel(entry.clientId);
    else await jobs.cancelJob(host.id, entry.id);
    toasts.push("Cancelled");
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

// Cancel-all is destructive → two-step inline confirm per host.
const cancelAllPendingId = ref<string | null>(null);
async function cancelAllOn(host: HostView) {
  if (cancelAllPendingId.value !== host.id) {
    cancelAllPendingId.value = host.id;
    return;
  }
  cancelAllPendingId.value = null;
  try {
    await jobs.cancelAll(host.id);
    toasts.push(`Cancelled queued jobs on ${host.label}`);
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

async function togglePause(host: HostView) {
  const snapshot = jobs.queues[host.id];
  try {
    if (snapshot?.paused) {
      await jobs.resume(host.id);
      toasts.push(`Queue resumed on ${host.label}`);
    } else {
      await jobs.pause(host.id);
      toasts.push(`Queue paused on ${host.label} — running job finishes`);
    }
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

/** Finished jobs from this session, freshest first. */
const finished = computed(() =>
  generation.jobs
    .filter((j) => j.status === "complete" || j.status === "error")
    .slice()
    .reverse(),
);

function finishedCode(job: Job): string {
  if (job.status === "complete") return "FIXED";
  return job.error === "Cancelled" ? "CANCELLED" : "STOPPED";
}

function reuse(job: Job) {
  composer.set({
    prompt: job.prompt,
    model: job.model,
    seed: null,
    width: job.width,
    height: job.height,
    steps: job.total,
    guidance: job.guidance,
  });
  void router.push("/generate");
}
</script>

<template>
  <div class="h-full overflow-y-auto p-6">
    <div class="mx-auto max-w-3xl">
      <div class="flex items-center gap-3">
        <h1 class="font-display text-display-md font-bold text-ink" style="font-stretch: 90%">
          Jobs
        </h1>
        <div class="flex-1" />
        <button
          v-if="finished.length"
          type="button"
          class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
          @click="generation.prune(0)"
        >
          Clear finished
        </button>
      </div>

      <!-- One queue section per connected host -->
      <section v-for="host in hosts.all" :key="host.id" data-test="host-queue" class="mt-6">
        <div class="flex items-center gap-2">
          <span class="edge-code">{{ host.label }}</span>
          <span
            v-if="jobs.queues[host.id]?.paused"
            class="edge-code text-safelight"
            data-test="paused-chip"
          >
            PAUSED
          </span>
          <div class="border-edge h-px flex-1 border-t" />
          <span v-if="host.queueDepth !== null" class="edge-code">
            {{ host.queueDepth
            }}<template v-if="host.queueCapacity">/{{ host.queueCapacity }}</template>
          </span>
          <button
            v-if="jobs.queues[host.id]?.caps?.canPause"
            type="button"
            data-test="pause-toggle"
            class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
            @click="togglePause(host)"
          >
            {{ jobs.queues[host.id]?.paused ? "Resume" : "Pause" }}
          </button>
          <button
            v-if="jobs.queues[host.id]?.caps?.canCancelAll && hostHasEntries(host)"
            type="button"
            data-test="cancel-all"
            class="h-7 rounded-control px-2.5 text-body"
            :class="cancelAllPendingId === host.id ? 'text-stop' : 'text-ink-3 hover:text-ink-2'"
            @click="cancelAllOn(host)"
            @blur="cancelAllPendingId = null"
          >
            {{ cancelAllPendingId === host.id ? "Cancel all?" : "Cancel all" }}
          </button>
        </div>

        <p v-if="jobs.queues[host.id]?.error" class="mt-2 text-caption text-stop">
          {{ jobs.queues[host.id]?.error }}
        </p>

        <!-- Multi-GPU hosts split into per-GPU lanes (drag a queued row between
             lanes, or right-click → Move to GPU N); single-GPU hosts keep the
             flat list below unchanged. -->
        <template v-if="hostHasEntries(host)">
          <div
            v-for="lane in hostLanes(host)"
            :key="lane.key"
            :data-test="lane.key === 'queue' ? 'queue-flat' : `gpu-lane-${lane.key}`"
            class="rounded-control border"
            :class="
              laneDroppable(host, lane.key) ? 'border-edge border-dashed' : 'border-transparent'
            "
            @dragover.prevent
            @drop.prevent="onLaneDrop(host, lane.key)"
          >
            <div v-if="lane.key !== 'queue'" class="mt-3 flex items-center gap-2">
              <span class="edge-code text-ink-3">GPU {{ lane.key }}</span>
              <div class="border-edge h-px flex-1 border-t" />
            </div>
            <ul v-if="lane.entries.length" class="mt-2 space-y-1.5">
              <li
                v-for="entry in lane.entries"
                :key="entry.id"
                data-test="queue-row"
                class="border-edge flex items-center gap-3 rounded-control border bg-bench px-3 py-2"
                :class="canDragEntry(host, entry) ? 'cursor-grab active:cursor-grabbing' : ''"
                :draggable="canDragEntry(host, entry)"
                @dragstart="onDragStart(host, entry, $event)"
                @dragend="dragged = null"
                @contextmenu="openEntryMenu(host, entry, $event)"
              >
                <div
                  class="h-12 w-12 shrink-0 overflow-hidden rounded-media border border-[color-mix(in_srgb,var(--rebate)_14%,transparent)] bg-print-surface"
                >
                  <img
                    v-if="ownJob(entry)?.previewUrl"
                    :src="ownJob(entry)!.previewUrl!"
                    alt=""
                    class="h-full w-full object-cover"
                    style="filter: blur(1px)"
                  />
                  <DevelopCanvas
                    v-else
                    :seed="ownJob(entry)?.visualSeed ?? entry.id"
                    :progress="ownJob(entry) ? jobProgress(ownJob(entry)!) : 0.2"
                    :phase="ownJob(entry) ? jobPhase(ownJob(entry)!) : 'latent'"
                  />
                </div>
                <div class="min-w-0 flex-1">
                  <div class="truncate text-body text-ink" :title="ownJob(entry)?.prompt">
                    {{ ownJob(entry)?.prompt ?? entry.model }}
                  </div>
                  <div class="mt-0.5 flex items-center gap-2">
                    <span class="edge-code">{{ entryCode(entry) }}</span>
                    <span
                      v-if="
                        lane.key !== 'queue' &&
                        entry.state === 'queued' &&
                        laneForEntry(entry) === null
                      "
                      class="edge-code text-ink-3"
                      title="No GPU requested — the server picks"
                    >
                      AUTO
                    </span>
                    <span class="text-caption text-ink-3">{{ entry.model }}</span>
                    <span v-if="!entry.mine" class="edge-code text-ink-3">OTHER CLIENT</span>
                  </div>
                </div>
                <button
                  v-if="entry.state === 'queued' || entry.mine"
                  type="button"
                  data-test="cancel-entry"
                  class="h-7 shrink-0 rounded-control px-2.5 text-body text-ink-3 hover:text-stop"
                  @click="cancelEntry(host, entry)"
                >
                  Cancel
                </button>
              </li>
            </ul>
            <p
              v-else-if="lane.key !== 'queue'"
              class="mt-2 px-3 py-2 text-caption text-ink-3"
              data-test="empty-lane"
            >
              Nothing on GPU {{ lane.key }}
            </p>
          </div>
        </template>
        <p v-else class="mt-2 text-caption text-ink-3">Nothing queued</p>
      </section>

      <!-- This session's finished prints -->
      <section v-if="finished.length" class="mt-8">
        <div class="flex items-center gap-2">
          <span class="edge-code">Finished this session</span>
          <div class="border-edge h-px flex-1 border-t" />
        </div>
        <ul class="mt-2 space-y-1.5">
          <li
            v-for="job in finished"
            :key="job.clientId"
            data-test="finished-row"
            class="border-edge flex items-center gap-3 rounded-control border bg-bench px-3 py-2"
          >
            <div
              class="h-12 w-12 shrink-0 overflow-hidden rounded-media border border-[color-mix(in_srgb,var(--rebate)_14%,transparent)] bg-print-surface"
            >
              <img
                v-if="job.resultUrl && !job.result?.video_frames"
                :src="job.resultUrl"
                alt=""
                class="h-full w-full object-cover"
              />
              <DevelopCanvas
                v-else
                :seed="job.visualSeed"
                :progress="jobProgress(job)"
                :phase="jobPhase(job)"
              />
            </div>
            <div class="min-w-0 flex-1">
              <div class="truncate text-body text-ink" :title="job.prompt">{{ job.prompt }}</div>
              <div class="mt-0.5 flex items-center gap-2">
                <span class="edge-code" :class="job.status === 'error' ? 'text-stop' : ''">
                  {{ finishedCode(job) }}
                </span>
                <span class="text-caption text-ink-3">
                  {{ job.model }}<template v-if="job.hostLabel"> · {{ job.hostLabel }}</template>
                </span>
                <span class="data-mono text-caption text-ink-3">S {{ job.visualSeed }}</span>
              </div>
            </div>
            <button
              type="button"
              class="border-edge h-7 shrink-0 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
              @click="reuse(job)"
            >
              Reuse
            </button>
          </li>
        </ul>
      </section>
    </div>
  </div>
</template>
