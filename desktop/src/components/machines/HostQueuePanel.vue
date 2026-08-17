<script setup lang="ts">
/*
 * One host's live server queue, with management — per-GPU lanes, drag/drop
 * (and a right-click "Move to GPU N" fallback) between lanes on multi-GPU
 * hosts, pause/resume, a two-step cancel-all, per-row cancel, and a row-click
 * info drawer that can push a job's settings back into Generate. Shared by the
 * standalone Jobs view and the Machines host-detail page so both surfaces
 * manage the queue identically. Single-GPU hosts keep the flat list.
 */
import { computed, ref, watch } from "vue";
import { useRouter } from "vue-router";
import DevelopCanvas from "@ui/components/DevelopCanvas.vue";
import QueueEntryDrawer from "../jobs/QueueEntryDrawer.vue";
import { useGenerationStore, jobPhase, jobProgress, type Job } from "../../stores/generation";
import { type HostView } from "../../stores/hosts";
import { enrichQueueEntries, useJobsStore, type EnrichedQueueEntry } from "../../stores/jobs";
import { useHostsStore } from "../../stores/hosts";
import { useHostModelsStore } from "../../stores/hostModels";
import { useComposerStore } from "../../stores/composer";
import { useToastStore } from "../../stores/toasts";
import { useContextMenuStore, type MenuEntry } from "../../stores/contextMenu";
import { formatEta } from "../../lib/format";
import { modelDisplayNameForId } from "../../lib/models";
import {
  computeQueueLanes,
  laneForEntry,
  resolveDropAction,
  type DraggedQueueRow,
  type LaneKey,
} from "../../lib/queueLanes";
import type { OutputMetadata } from "../../lib/api/types";

const props = withDefaults(
  defineProps<{
    host: HostView;
    /** Row `data-test` id — differs between the Jobs view and host detail. */
    rowTestId?: string;
    /** Line shown when the host has nothing queued. */
    emptyLabel?: string;
    /** Show the Develop-canvas thumbnail beside each row. */
    thumbnails?: boolean;
    /** Show the pause/resume + cancel-all controls row. */
    controls?: boolean;
  }>(),
  {
    rowTestId: "queue-row",
    emptyLabel: "Nothing queued",
    thumbnails: true,
    controls: true,
  },
);

const router = useRouter();
const generation = useGenerationStore();
const hosts = useHostsStore();
const hostModels = useHostModelsStore();
const jobs = useJobsStore();
const composer = useComposerStore();
const toasts = useToastStore();
const contextMenu = useContextMenuStore();

const primaryId = computed(() => hosts.primaryHost?.id ?? "local");
const snapshot = computed(() => jobs.queues[props.host.id] ?? null);
const paused = computed(() => snapshot.value?.paused === true);
const caps = computed(() => snapshot.value?.caps ?? null);

const lanes = computed(() => {
  const snap = snapshot.value;
  const entries = snap
    ? enrichQueueEntries(snap.entries, props.host.id, generation.jobs, primaryId.value)
    : [];
  return computeQueueLanes(entries, snap?.gpuOrdinals ?? []);
});

const hasEntries = computed(() => lanes.value.some((lane) => lane.entries.length > 0));
const modelLabel = (name: string) =>
  modelDisplayNameForId(name, hostModels.modelsOn(props.host.id));

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
  // Held work is parked, not in line. Showing it a position tells the operator
  // to wait for something the host will never start on its own.
  if (entry.state === "held") return "HELD";
  return entry.position > 0 ? `QUEUED #${entry.position}` : "QUEUED";
}

/** The host's own words for why a job is parked, shown beside the row so the
 *  operator can act on it rather than just seeing it stuck. */
function entryHeldReason(entry: EnrichedQueueEntry): string | null {
  if (entry.state !== "held") return null;
  return entry.held_reason?.trim() || null;
}

/** Elapsed wall-clock for running entries; re-evaluates on each poll frame. */
function entryElapsed(entry: EnrichedQueueEntry): string | null {
  if (entry.state !== "running" || !entry.started_at_unix_ms) return null;
  return formatEta((Date.now() - entry.started_at_unix_ms) / 1000);
}

// ── Per-GPU lanes (multi-GPU hosts only) ─────────────────────────────────
const dragged = ref<DraggedQueueRow | null>(null);

function laneOrdinals(): number[] {
  return lanes.value.map((l) => l.key).filter((k): k is number => k !== "queue");
}

function canDragEntry(entry: EnrichedQueueEntry): boolean {
  return entry.state === "queued" && laneOrdinals().length > 0;
}

function onDragStart(entry: EnrichedQueueEntry, event: DragEvent) {
  if (!canDragEntry(entry)) {
    dragged.value = null;
    return;
  }
  dragged.value = { hostId: props.host.id, entryId: entry.id, lane: laneForEntry(entry) };
  event.dataTransfer?.setData("text/plain", entry.id);
  if (event.dataTransfer) event.dataTransfer.effectAllowed = "move";
}

function onLaneDrop(laneKey: LaneKey) {
  const action = resolveDropAction(dragged.value, props.host.id, laneKey);
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
function openEntryMenu(entry: EnrichedQueueEntry, event: MouseEvent) {
  const ordinals = laneOrdinals();
  if (entry.state !== "queued" || ordinals.length === 0) return;
  const current = laneForEntry(entry);
  const items: MenuEntry[] = ordinals.map((ordinal) => ({
    label: `Move to GPU ${ordinal}`,
    disabled: ordinal === current,
    action: () => void jobs.reassignGpu(props.host.id, entry.id, ordinal),
  }));
  contextMenu.open(event, items);
}

/** Highlight a lane while a same-host drag could land on it. */
function laneDroppable(laneKey: LaneKey): boolean {
  return (
    dragged.value !== null &&
    dragged.value.hostId === props.host.id &&
    laneKey !== "queue" &&
    laneKey !== dragged.value.lane
  );
}

async function cancelEntry(entry: EnrichedQueueEntry) {
  try {
    const cancelled =
      entry.clientId !== null
        ? await generation.cancel(entry.clientId)
        : await jobs.cancelJob(props.host.id, entry.id).then(() => true);
    if (cancelled) toasts.push("Cancelled");
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

// Cancel-all is destructive → two-step inline confirm.
const cancelAllPending = ref(false);
async function cancelAll() {
  if (!cancelAllPending.value) {
    cancelAllPending.value = true;
    return;
  }
  cancelAllPending.value = false;
  try {
    await jobs.cancelAll(props.host.id);
    toasts.push(`Cancelled queued jobs on ${props.host.label}`);
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

async function togglePause() {
  try {
    if (paused.value) {
      await jobs.resume(props.host.id);
      toasts.push(`Queue resumed on ${props.host.label}`);
    } else {
      await jobs.pause(props.host.id);
      toasts.push(`Queue paused on ${props.host.label} — running job finishes`);
    }
  } catch (err) {
    toasts.push(String(err), "error");
  }
}

// ── Row info drawer (state + submitted settings → Generate) ───────────────
const queueDetail = ref<EnrichedQueueEntry | null>(null);

// Each poll rebuilds the entry objects — keep an open drawer tracking its job
// by id so state/elapsed stay live.
const flatEntries = computed(() => lanes.value.flatMap((lane) => lane.entries));
watch(flatEntries, (entries) => {
  const open = queueDetail.value;
  if (!open) return;
  const updated = entries.find((entry) => entry.id === open.id);
  queueDetail.value = updated ?? null;
});

/** Unpinned seeds restore as random; `seed_pinned` disambiguates seed 0. */
function loadQueueSettings(metadata: OutputMetadata) {
  const pinned = queueDetail.value?.seed_pinned ?? metadata.seed !== 0;
  composer.set({
    metadata: pinned ? metadata : ({ ...metadata, seed: null } as unknown as OutputMetadata),
  });
  queueDetail.value = null;
  void router.push("/generate");
}
</script>

<template>
  <div>
    <!-- Management controls -->
    <div
      v-if="controls && (caps?.canPause || (caps?.canCancelAll && hasEntries) || paused)"
      class="mb-2 flex items-center gap-2"
    >
      <span v-if="paused" data-test="paused-chip" class="edge-code text-safelight">PAUSED</span>
      <div class="flex-1" />
      <button
        v-if="caps?.canPause"
        type="button"
        data-test="pause-toggle"
        class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
        @click="togglePause"
      >
        {{ paused ? "Resume" : "Pause" }}
      </button>
      <button
        v-if="caps?.canCancelAll && hasEntries"
        type="button"
        data-test="cancel-all"
        class="h-7 rounded-control px-2.5 text-body"
        :class="cancelAllPending ? 'text-stop' : 'text-ink-3 hover:text-ink-2'"
        @click="cancelAll"
        @blur="cancelAllPending = false"
      >
        {{ cancelAllPending ? "Cancel all?" : "Cancel all" }}
      </button>
    </div>

    <p v-if="snapshot?.error" class="mt-1 text-caption text-stop">{{ snapshot.error }}</p>

    <!-- Multi-GPU hosts split into per-GPU lanes; single-GPU hosts stay flat. -->
    <template v-if="hasEntries">
      <div
        v-for="lane in lanes"
        :key="lane.key"
        :data-test="lane.key === 'queue' ? 'queue-flat' : `gpu-lane-${lane.key}`"
        class="rounded-control border"
        :class="laneDroppable(lane.key) ? 'border-edge border-dashed' : 'border-transparent'"
        @dragover.prevent
        @drop.prevent="onLaneDrop(lane.key)"
      >
        <div v-if="lane.key !== 'queue'" class="mt-3 flex items-center gap-2">
          <span class="edge-code text-ink-3">GPU {{ lane.key }}</span>
          <div class="border-edge h-px flex-1 border-t" />
        </div>
        <ul v-if="lane.entries.length" class="mt-2 space-y-1.5">
          <li
            v-for="entry in lane.entries"
            :key="entry.id"
            :data-test="rowTestId"
            role="button"
            tabindex="0"
            class="border-edge flex cursor-pointer items-center gap-3 rounded-control border bg-bench px-3 py-2 transition-colors hover:bg-bath"
            :class="canDragEntry(entry) ? 'active:cursor-grabbing' : ''"
            :draggable="canDragEntry(entry)"
            :aria-label="`Show details for ${modelLabel(entry.model)}`"
            @dragstart="onDragStart(entry, $event)"
            @dragend="dragged = null"
            @contextmenu="openEntryMenu(entry, $event)"
            @click="queueDetail = entry"
            @keydown.enter="queueDetail = entry"
          >
            <div
              v-if="thumbnails"
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
            <span
              v-else
              class="h-1.5 w-1.5 shrink-0 rounded-full"
              :class="entry.state === 'running' ? 'bg-safelight' : 'bg-halide'"
              aria-hidden="true"
            />
            <div class="min-w-0 flex-1">
              <div class="truncate text-body text-ink" :title="ownJob(entry)?.prompt">
                {{ ownJob(entry)?.prompt ?? modelLabel(entry.model) }}
              </div>
              <div class="mt-0.5 flex items-center gap-2">
                <span class="edge-code">{{ entryCode(entry) }}</span>
                <span
                  v-if="
                    lane.key !== 'queue' && entry.state === 'queued' && laneForEntry(entry) === null
                  "
                  class="edge-code text-ink-3"
                  title="No GPU requested — the server picks"
                >
                  AUTO
                </span>
                <span class="truncate text-caption text-ink-3">{{ modelLabel(entry.model) }}</span>
                <span
                  v-if="entryHeldReason(entry)"
                  class="truncate text-caption text-ink-3"
                  :title="entryHeldReason(entry) ?? undefined"
                >
                  · {{ entryHeldReason(entry) }}
                </span>
                <span v-if="!entry.mine" class="edge-code shrink-0 text-ink-3">OTHER CLIENT</span>
                <span v-if="entryElapsed(entry)" class="data-mono shrink-0 text-ink-3">
                  {{ entryElapsed(entry) }}
                </span>
              </div>
            </div>
            <button
              v-if="entry.state === 'queued' || entry.state === 'held' || entry.mine"
              type="button"
              data-test="cancel-entry"
              class="h-7 shrink-0 rounded-control px-2.5 text-body text-ink-3 hover:text-stop"
              @click.stop="cancelEntry(entry)"
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
    <p v-else class="mt-1 text-caption text-ink-3" data-test="queue-empty">{{ emptyLabel }}</p>

    <QueueEntryDrawer
      v-if="queueDetail"
      :entry="queueDetail"
      :model-label="modelLabel(queueDetail.model)"
      :host-label="host.label"
      :state-code="entryCode(queueDetail)"
      :elapsed="entryElapsed(queueDetail)"
      @close="queueDetail = null"
      @load="loadQueueSettings"
    />
  </div>
</template>
