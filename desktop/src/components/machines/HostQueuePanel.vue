<script setup lang="ts">
/*
 * One host's live server queue, with management — per-GPU lanes, drag/drop
 * (and a right-click "Move to GPU N" fallback) between lanes on multi-GPU
 * hosts, pause/resume, a two-step cancel-all, per-row cancel, and a row-click
 * info drawer that can push a job's settings back into Generate. Shared by the
 * standalone Jobs view and the Machines host-detail page so both surfaces
 * manage the queue identically. Single-GPU hosts keep the flat list.
 */
import { computed, onUnmounted, ref, watch } from "vue";
import { useRouter } from "vue-router";
import DevelopCanvas from "@ui/components/DevelopCanvas.vue";
import QueueEntryDrawer from "../jobs/QueueEntryDrawer.vue";
import ConfirmDialog from "../shell/ConfirmDialog.vue";
import { useGenerationStore, jobPhase, jobProgress, type Job } from "../../stores/generation";
import { type HostView } from "../../stores/hosts";
import {
  buildQueueStatusIndex,
  queueRuntimeCode,
  queueStatusFor,
  queueWaitCode,
  resolveQueueWait,
} from "@studio/lib/queuePosition";
import { queueEntryDetailModel, type QueueDetailMetadata } from "@studio/lib/queueEntryDetail";
import { watchSelectedQueuePreview, type QueueJobProgress } from "@studio/api/generationSelection";
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
import QueuePlanWorkList from "@studio/components/QueuePlanWorkList.vue";
import { queuePlanOnlyWork } from "@studio/lib/queuePlanPresentation";
import { settingsRestoreMetadata } from "@studio/api/generationSelection";

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
const cancellingIds = ref<string[]>([]);
const pausingIds = ref<string[]>([]);
const retryingIds = ref<string[]>([]);
const composer = useComposerStore();
const toasts = useToastStore();
const contextMenu = useContextMenuStore();

const primaryId = computed(() => hosts.primaryHost?.id ?? "local");
const snapshot = computed(() => jobs.queues[props.host.id] ?? null);
const dispatchPaused = computed(() => snapshot.value?.paused === true);
const resumeNeeded = computed(() => dispatchPaused.value);
const caps = computed(() => snapshot.value?.caps ?? null);
const queueStatus = computed(() => {
  const current = snapshot.value;
  return buildQueueStatusIndex([
    {
      hostId: props.host.id,
      entries: current?.entries ?? [],
      plan: current?.plan ?? null,
      paused: current?.paused,
    },
  ]);
});

const lanes = computed(() => {
  const snap = snapshot.value;
  const entries = snap
    ? enrichQueueEntries(
        snap.entries,
        props.host.id,
        generation.jobs,
        primaryId.value,
        (clientBatchId) => generation.ownsDurableBatch(clientBatchId),
      )
    : [];
  return computeQueueLanes(entries, snap?.gpuOrdinals ?? []);
});

const hasEntries = computed(() => lanes.value.some((lane) => lane.entries.length > 0));
const entryIds = computed(() =>
  lanes.value.flatMap((lane) => lane.entries.map((entry) => entry.id)),
);
const hasPlanOnlyWork = computed(
  () => queuePlanOnlyWork(snapshot.value?.plan, entryIds.value).length > 0,
);
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
    const runtime = queueRuntimeCode(snapshot.value?.plan, entry.id);
    if (runtime) return runtime;
    return entry.gpu !== undefined ? `RUNNING · GPU ${entry.gpu}` : "RUNNING";
  }
  if (dispatchPaused.value && entry.state === "queued") return "PAUSED";
  // Same waiting vocabulary as Create and iPhone, resolved once in studio —
  // including "held", which is parked rather than in line.
  return queueWaitCode(
    resolveQueueWait(
      queueStatusFor(queueStatus.value, props.host.id, entry.id) ?? {
        state: entry.state,
        position: entry.position,
      },
    ),
  );
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

/** Queue actions live on every row. GPU reassignment is appended when the
 * selected queued row has real device lanes. */
function openEntryMenu(entry: EnrichedQueueEntry, event: MouseEvent) {
  const ordinals = laneOrdinals();
  const items: MenuEntry[] = [
    { label: "Show details", action: () => void openQueueDetail(entry) },
    { label: "Reuse settings", action: () => void reuseEntry(entry) },
    { separator: true },
  ];
  if (caps.value?.canPauseJob && (entry.state === "queued" || entry.state === "paused")) {
    items.push({
      label: entry.state === "paused" ? "Resume job" : "Pause job",
      action: () => void toggleEntryPause(entry),
    });
  }
  if (entry.state === "held" && entry.retryable === true) {
    items.push({ label: "Retry job", action: () => void retryFromMenu(entry) });
  }
  items.push({
    label: entry.state === "running" ? "Stop job" : "Cancel job",
    danger: true,
    disabled: entry.state === "running" && caps.value?.canCancelRunning !== true,
    action: () => {
      confirmCancel.value = entry;
    },
  });
  if (entry.state === "queued" && ordinals.length > 0) {
    const current = laneForEntry(entry);
    items.push(
      { separator: true },
      ...ordinals.map((ordinal) => ({
        label: `Move to GPU ${ordinal}`,
        disabled: ordinal === current,
        action: () => void jobs.reassignGpu(props.host.id, entry.id, ordinal),
      })),
    );
  }
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
  if (cancellingIds.value.includes(entry.id)) return;
  cancellingIds.value = [...cancellingIds.value, entry.id];
  try {
    const cancelled =
      entry.clientId !== null
        ? await generation.cancel(entry.clientId)
        : await jobs.cancelJob(props.host.id, entry.id).then(() => true);
    if (cancelled) toasts.push("Cancelled");
  } catch (err) {
    toasts.push(String(err), "error");
  } finally {
    cancellingIds.value = cancellingIds.value.filter((id) => id !== entry.id);
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
    if (resumeNeeded.value) {
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

async function toggleEntryPause(entry: EnrichedQueueEntry) {
  if (pausingIds.value.includes(entry.id)) return;
  pausingIds.value = [...pausingIds.value, entry.id];
  const paused = entry.state !== "paused";
  try {
    await jobs.setJobPaused(props.host.id, entry.id, paused);
    toasts.push(`${paused ? "Paused" : "Resumed"} job on ${props.host.label}`);
  } catch (err) {
    toasts.push(String(err), "error");
  } finally {
    pausingIds.value = pausingIds.value.filter((id) => id !== entry.id);
  }
}

// ── Row info drawer (everything the host says about one queued job) ───────
const queueDetail = ref<EnrichedQueueEntry | null>(null);
const detailError = ref<string | null>(null);
let detailRequestVersion = 0;
const detailPreview = ref<QueueJobProgress | null>(null);
const detailNowMs = ref(Date.now());
let stopPreview: (() => void) | null = null;
let nowTimer: ReturnType<typeof setInterval> | null = null;

// Each poll rebuilds the entry objects — keep an open drawer tracking its job
// by id so state/elapsed stay live.
const flatEntries = computed(() => lanes.value.flatMap((lane) => lane.entries));
watch(flatEntries, (entries) => {
  const open = queueDetail.value;
  if (!open) return;
  const updated = entries.find((entry) => entry.id === open.id);
  queueDetail.value = updated
    ? {
        ...open,
        ...updated,
        // Hot queue polling is intentionally payload-free. Once an explicit
        // row read hydrates settings, keep them while merging live state.
        ...(updated.metadata !== undefined
          ? { metadata: updated.metadata }
          : open.metadata !== undefined
            ? { metadata: open.metadata }
            : {}),
        ...(updated.batch_id !== undefined
          ? { batch_id: updated.batch_id }
          : open.batch_id !== undefined
            ? { batch_id: open.batch_id }
            : {}),
        ...(updated.client_batch_id !== undefined
          ? { client_batch_id: updated.client_batch_id }
          : open.client_batch_id !== undefined
            ? { client_batch_id: open.client_batch_id }
            : {}),
        ...(updated.batch_index !== undefined
          ? { batch_index: updated.batch_index }
          : open.batch_index !== undefined
            ? { batch_index: open.batch_index }
            : {}),
      }
    : null;
});

function closeDetail(): void {
  detailRequestVersion += 1;
  queueDetail.value = null;
}

async function hydratedEntry(entry: EnrichedQueueEntry): Promise<EnrichedQueueEntry> {
  try {
    const job = await jobs.queueJob(props.host.id, entry.id);
    return {
      ...entry,
      ...job,
      mine: entry.mine || generation.ownsDurableBatch(job.client_batch_id),
    };
  } catch (error) {
    // Older hosts have no single-job detail route. A row already owned by
    // this app still has its request and retry authority in local recovery.
    if (ownJob(entry)) return entry;
    throw error;
  }
}

async function openQueueDetail(entry: EnrichedQueueEntry): Promise<void> {
  const requestVersion = ++detailRequestVersion;
  detailError.value = null;
  // Open synchronously so keyboard/click activation preserves the existing
  // desktop interaction even while the authoritative row is fetched.
  queueDetail.value = entry;
  try {
    const hydrated = await hydratedEntry(entry);
    if (requestVersion === detailRequestVersion && queueDetail.value?.id === entry.id) {
      queueDetail.value = hydrated;
    }
  } catch (error) {
    if (requestVersion === detailRequestVersion && queueDetail.value?.id === entry.id) {
      detailError.value = error instanceof Error ? error.message : String(error);
    }
  }
}

/** A row this app submitted carries its exact request here, which is the one
 *  thing that closes the durable listing's payload-free window client-side. */
function localMetadataFor(entry: EnrichedQueueEntry): QueueDetailMetadata | null {
  return (ownJob(entry)?.request as QueueDetailMetadata | undefined) ?? null;
}

/** Current servers expose durable retry authority on the selected row. Keep
 * this app's in-memory authority as a compatibility path for older listings. */
function retryAuthorityFor(entry: EnrichedQueueEntry): unknown | null {
  if (
    entry.state === "held" &&
    entry.retryable === true &&
    entry.batch_id &&
    entry.client_batch_id
  ) {
    return entry;
  }
  const job = ownJob(entry);
  return job && job.retryable && !job.retrying ? job.clientId : null;
}

const queueDetailModel = computed(() => {
  const entry = queueDetail.value;
  if (!entry) return null;
  return queueEntryDetailModel({
    entry,
    hostLabel: props.host.label,
    modelLabel: modelLabel(entry.model),
    nowMs: detailNowMs.value,
    plan: snapshot.value?.plan ?? null,
    metadata: (entry.metadata as QueueDetailMetadata | null | undefined) ?? null,
    localMetadata: localMetadataFor(entry),
    mine: entry.mine,
    canCancelRunning: caps.value?.canCancelRunning === true,
    retryAuthority: retryAuthorityFor(entry),
  });
});

// The drawer's elapsed and estimate lines are wall-clock, so they need their
// own tick — the 5 s queue poll is far too coarse to watch a job run.
watch(
  () => queueDetail.value?.id ?? null,
  (id) => {
    detailError.value = null;
    detailPreview.value = null;
    stopPreview?.();
    stopPreview = null;
    if (nowTimer !== null) clearInterval(nowTimer);
    nowTimer = null;
    if (id === null) return;
    detailNowMs.value = Date.now();
    nowTimer = setInterval(() => (detailNowMs.value = Date.now()), 1_000);

    const entry = queueDetail.value;
    const target = jobs.targetFor(props.host);
    if (!entry || entry.state !== "running" || !target) return;
    stopPreview = watchSelectedQueuePreview(
      target,
      id,
      (preview) => (detailPreview.value = preview),
      750,
      () => (detailPreview.value = null),
    );
  },
);

onUnmounted(() => {
  stopPreview?.();
  if (nowTimer !== null) clearInterval(nowTimer);
});

/** Unpinned seeds restore as random; `seed_pinned` disambiguates seed 0. */
function loadQueueSettings(metadata: OutputMetadata) {
  const detail = queueDetail.value;
  const pinned = detail?.seed_pinned ?? metadata.seed !== 0;
  const restored = settingsRestoreMetadata(metadata, { seedPinned: pinned });
  composer.set(
    detail
      ? {
          metadata: restored,
          queueSelection: {
            hostId: props.host.id,
            jobId: detail.id,
            running: detail.state === "running",
          },
        }
      : { metadata: restored },
  );
  queueDetail.value = null;
  void router.push("/create");
}

/** Reuse takes whichever settings the model resolved — the host's own, or
 *  this app's copy of the request it submitted. */
function reuseFromDrawer(): void {
  const entry = queueDetail.value;
  if (!entry) return;
  const metadata =
    (entry.metadata as OutputMetadata | null | undefined) ??
    (localMetadataFor(entry) as OutputMetadata | null) ??
    null;
  if (!metadata) return;
  loadQueueSettings(metadata);
}

async function reuseEntry(entry: EnrichedQueueEntry): Promise<void> {
  try {
    const hydrated = await hydratedEntry(entry);
    const metadata =
      (hydrated.metadata as OutputMetadata | null | undefined) ??
      (localMetadataFor(hydrated) as OutputMetadata | null);
    if (!metadata) throw new Error("The machine did not return settings for this job.");
    queueDetail.value = hydrated;
    loadQueueSettings(metadata);
  } catch (error) {
    toasts.push(error instanceof Error ? error.message : String(error), "error");
  }
}

// Cancelling from the drawer is destructive and names the job, so it takes the
// shared plain ConfirmDialog rather than the row's inline two-step.
const confirmCancel = ref<EnrichedQueueEntry | null>(null);

function cancelFromDrawer(): void {
  const entry = queueDetail.value;
  if (!entry) return;
  detailError.value = null;
  confirmCancel.value = entry;
}

async function cancelConfirmed(): Promise<void> {
  const entry = confirmCancel.value;
  confirmCancel.value = null;
  if (!entry) return;
  try {
    await cancelEntry(entry);
    queueDetail.value = null;
  } catch (error) {
    detailError.value = error instanceof Error ? error.message : String(error);
  }
}

async function retryFromDrawer(): Promise<void> {
  const entry = queueDetail.value;
  if (!entry) return;
  detailError.value = null;
  try {
    await retryEntry(entry);
  } catch (error) {
    detailError.value = error instanceof Error ? error.message : String(error);
  }
}

async function retryEntry(entry: EnrichedQueueEntry): Promise<void> {
  if (retryingIds.value.includes(entry.id)) return;
  retryingIds.value = [...retryingIds.value, entry.id];
  try {
    const hydrated = await hydratedEntry(entry);
    if (hydrated.batch_id && hydrated.client_batch_id) {
      await jobs.retryJob(props.host.id, hydrated);
    } else {
      const clientId = ownJob(hydrated)?.clientId ?? null;
      if (clientId === null) throw new Error("This held generation is not retryable.");
      await generation.retryHeld(clientId);
    }
    toasts.push("Retry queued");
    if (queueDetail.value?.id === entry.id) queueDetail.value = null;
  } finally {
    retryingIds.value = retryingIds.value.filter((id) => id !== entry.id);
  }
}

async function retryFromMenu(entry: EnrichedQueueEntry): Promise<void> {
  try {
    await retryEntry(entry);
  } catch (error) {
    toasts.push(error instanceof Error ? error.message : String(error), "error");
  }
}
</script>

<template>
  <div>
    <!-- Management controls -->
    <div
      v-if="controls && (caps?.canPause || (caps?.canCancelAll && hasEntries) || resumeNeeded)"
      class="mb-2 flex items-center gap-2"
    >
      <span v-if="resumeNeeded" data-test="paused-chip" class="edge-code text-safelight">
        PAUSED
      </span>
      <div class="flex-1" />
      <button
        v-if="caps?.canPause"
        type="button"
        data-test="pause-toggle"
        class="border-edge h-7 rounded-control border px-2.5 text-body text-ink-2 hover:text-ink"
        @click="togglePause"
      >
        {{ resumeNeeded ? "Resume" : "Pause" }}
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
            @click="openQueueDetail(entry)"
            @keydown.enter="openQueueDetail(entry)"
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
              v-if="caps?.canPauseJob && (entry.state === 'queued' || entry.state === 'paused')"
              type="button"
              data-test="pause-entry"
              :disabled="pausingIds.includes(entry.id)"
              class="h-7 shrink-0 rounded-control px-2.5 text-body text-ink-3 hover:text-ink"
              @click.stop="toggleEntryPause(entry)"
            >
              {{
                pausingIds.includes(entry.id)
                  ? "Working…"
                  : entry.state === "paused"
                    ? "Resume"
                    : "Pause"
              }}
            </button>
            <button
              v-if="
                entry.state === 'queued' ||
                entry.state === 'paused' ||
                entry.state === 'held' ||
                (entry.state === 'running' && caps?.canCancelRunning)
              "
              type="button"
              data-test="cancel-entry"
              :disabled="cancellingIds.includes(entry.id)"
              class="h-7 shrink-0 rounded-control px-2.5 text-body text-ink-3 hover:text-stop"
              @click.stop="cancelEntry(entry)"
            >
              {{ cancellingIds.includes(entry.id) ? "Cancelling…" : "Cancel" }}
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
    <QueuePlanWorkList class="mt-2" :plan="snapshot?.plan ?? null" :exclude-ids="entryIds" />
    <button
      v-if="snapshot?.nextCursor"
      type="button"
      data-test="queue-load-more"
      class="border-edge mt-2 h-8 w-full rounded-control border px-3 text-caption text-ink-2 hover:text-ink disabled:opacity-50"
      :disabled="snapshot.loadingMore"
      @click="jobs.loadMoreHost(host.id)"
    >
      {{ snapshot.loadingMore ? "Loading…" : "Load more jobs" }}
    </button>
    <p v-if="snapshot?.loadMoreError" class="mt-1 text-caption text-stop">
      {{ snapshot.loadMoreError }}
    </p>
    <p
      v-if="!hasEntries && !hasPlanOnlyWork"
      class="mt-1 text-caption text-ink-3"
      data-test="queue-empty"
    >
      {{ emptyLabel }}
    </p>

    <QueueEntryDrawer
      v-if="queueDetailModel"
      :model="queueDetailModel"
      :preview="detailPreview"
      :cancelling="queueDetail ? cancellingIds.includes(queueDetail.id) : false"
      :retrying="queueDetail ? retryingIds.includes(queueDetail.id) : false"
      :error="detailError"
      @close="closeDetail"
      @reuse="reuseFromDrawer"
      @cancel="cancelFromDrawer"
      @retry="retryFromDrawer"
    />

    <ConfirmDialog
      :open="confirmCancel !== null"
      :title="confirmCancel?.state === 'running' ? 'Stop this job?' : 'Cancel this job?'"
      :message="`${confirmCancel ? modelLabel(confirmCancel.model) : ''} on ${host.label}. ${
        confirmCancel?.state === 'running'
          ? 'The machine stops at its next safe point and nothing is saved.'
          : 'It leaves the queue and is not rendered.'
      }`"
      :confirm-label="confirmCancel?.state === 'running' ? 'Stop job' : 'Cancel job'"
      danger
      @confirm="cancelConfirmed"
      @cancel="confirmCancel = null"
    />
  </div>
</template>
