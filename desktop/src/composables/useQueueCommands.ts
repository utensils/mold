import {
  missingQueueModel,
  startQueueDownloadRecovery,
} from "@studio/composables/useQueueDownloadRecovery";
import { useHeldQueueTransfer } from "@studio/composables/useHeldQueueTransfer";
import { computed, ref, type ComputedRef, type Ref } from "vue";
import { useRouter } from "vue-router";
import { apiFetchTo, apiJsonTo } from "@studio/api/client";
import type { FleetActiveWork } from "@studio/api/activity";
import { useOpenLiveWork } from "./useOpenLiveWork";
import { useQueueActivity, type QueueRow } from "./useQueueActivity";
import { jobCanBeRemoved, useGenerationStore, type Job } from "../stores/generation";
import { useComposerStore } from "../stores/composer";
import { useContextMenuStore, type MenuEntry } from "../stores/contextMenu";
import { useHostsStore } from "../stores/hosts";
import { useHostStatusStore } from "../stores/hostStatus";
import { useJobsStore } from "../stores/jobs";
import { useLiveActivityStore } from "../stores/liveActivity";
import { useToastStore } from "../stores/toasts";

export interface QueueCommands {
  /** Whether the display host reports a pausable queue. */
  canPause: ComputedRef<boolean>;
  paused: ComputedRef<boolean>;
  togglePause(): Promise<void>;
  /** The same three questions for ONE named machine, which is what a queue row
   * needs: the rail's active card is fleet-wide, so the machine it shows is
   * not necessarily the one the status bar displays. */
  canPauseFor(hostId: string | null): boolean;
  pausedFor(hostId: string | null): boolean;
  togglePauseFor(hostId: string | null): Promise<void>;
  /** The machine a row's work runs on, whatever kind of row it is. */
  hostIdFor(row: QueueRow): string | null;
  /** Cancel every live print this client owns and every host's queue. The
   * unconfirmed primitive: every surface goes through `askStopEverything`. */
  stopEverything(): Promise<void>;
  /** Raise the one shared confirm. The action is fleet-wide and irreversible,
   * so no surface fires it directly. */
  askStopEverything(): void;
  confirmStopEverything(): Promise<void>;
  cancelStopEverything(): void;
  stopEverythingOpen: Ref<boolean>;
  stopEverythingBusy: Ref<boolean>;
  /** What the confirm says it is about to stop, in pictures and machines. */
  stopEverythingSummary: ComputedRef<string>;
  canCancel(row: QueueRow): boolean;
  cancel(row: QueueRow): Promise<void>;
  /** Whether this row may be dragged: its host offers reorder and the row is
   * one of that host's queued entries. */
  canReorder(row: QueueRow): boolean;
  /** Drop `dragged` onto `target`, taking the target's slot. Refuses a drop on
   * itself and across machines — a queue position only means something inside
   * one host's line. */
  dropOn(dragged: QueueRow, target: QueueRow): Promise<void>;
  /** Bring the row to the canvas (or the surface that can inspect it). */
  open(row: QueueRow): void;
  contextMenu(event: MouseEvent, row: QueueRow): void;
  cancellingShared: Ref<string[]>;
}

/*
 * Module scope, deliberately. `useQueueCommands` is instantiated per SURFACE
 * (rail, Queue view, status bar, palette, App) and per ROW — `QueueRowMenu` is
 * rendered once per row — so state that describes the APP rather than one
 * caller has to live above the factory:
 *
 *  - `cancellingShared` is the in-flight guard for a shared row's Stop. Held
 *    per instance, the Queue view's Stop stayed armed while the same row's
 *    ⋯ ▸ Stop had a request in the air, and both were sent.
 *  - the Stop-everything dialog is ONE dialog, rendered once in App.vue and
 *    opened from three doors (the rail, the Queue view, ⌘K).
 *
 * Pinia does not reset these between tests; `__resetQueueCommandState()` does.
 */
const cancellingShared = ref<string[]>([]);
const stopEverythingOpen = ref(false);
const stopEverythingBusy = ref(false);

/** Clear the module-scoped queue state. Tests only. */
export function __resetQueueCommandState(): void {
  cancellingShared.value = [];
  stopEverythingOpen.value = false;
  stopEverythingBusy.value = false;
}

/**
 * Every action the queue offers, shared by the sidebar rail and the Queue
 * view so both surfaces act on the same authorities.
 */
export function useQueueCommands(): QueueCommands {
  const router = useRouter();
  const transfer = useHeldQueueTransfer();
  const composer = useComposerStore();
  const contextMenu = useContextMenuStore();
  const generation = useGenerationStore();
  const hosts = useHostsStore();
  const hostStatus = useHostStatusStore();
  const jobs = useJobsStore();
  const liveActivity = useLiveActivityStore();
  const toasts = useToastStore();
  const queue = useQueueActivity();
  const openLiveWork = useOpenLiveWork();

  const displayQueue = computed(() => {
    const host = hostStatus.displayHost;
    return host ? (jobs.queues[host.id] ?? null) : null;
  });
  const canPause = computed(() => displayQueue.value?.caps?.canPause === true);
  const paused = computed(() => displayQueue.value?.paused === true);

  /** One named machine's queue snapshot, or null while nothing has read it. */
  function queueFor(hostId: string | null) {
    return hostId ? (jobs.queues[hostId] ?? null) : null;
  }
  function canPauseFor(hostId: string | null): boolean {
    return queueFor(hostId)?.caps?.canPause === true;
  }
  function pausedFor(hostId: string | null): boolean {
    return queueFor(hostId)?.paused === true;
  }

  function report(error: unknown) {
    toasts.push(error instanceof Error ? error.message : String(error), "error");
  }

  /**
   * Pause or resume ONE machine's queue. The snapshot is refreshed FIRST:
   * `paused` is read off `jobs.queues`, and pausing a host whose snapshot has
   * not been read yet writes nothing back — so a second Space would pause
   * again instead of resuming, leaving the queue stopped with no way back.
   *
   * The one thing that happens BEFORE that refresh is the capability check,
   * and only against a snapshot that has already been read: a machine known
   * not to pause costs no request at all, while a machine nobody has asked yet
   * still refreshes, because a cold launch must be able to pause.
   */
  async function togglePauseFor(hostId: string | null) {
    if (!hostId) return;
    const display = hostStatus.displayHost;
    const host =
      hosts.all.find((candidate) => candidate.id === hostId) ??
      (display?.id === hostId ? display : null);
    if (!host) return;
    if (queueFor(hostId) && !canPauseFor(hostId)) return;
    try {
      await jobs.refreshHost(host);
      if (!canPauseFor(hostId)) return;
      if (pausedFor(hostId)) await jobs.resume(hostId);
      else await jobs.pause(hostId);
    } catch (error) {
      report(error);
    }
  }

  /** Pause or resume the DISPLAY host's queue — the rail's header control,
   * the status bar's hint, and Space. */
  async function togglePause() {
    await togglePauseFor(hostStatus.displayHost?.id ?? null);
  }

  async function cancelPrint(job: Job) {
    try {
      if (await generation.cancel(job.clientId)) toasts.push("Stopped");
    } catch (error) {
      report(error);
    }
  }

  /**
   * The live route to a shared row's machine, or null when this client's view
   * of that machine has moved on. Acting on a stale snapshot would send the
   * request to whatever now answers on that URL, so both Stop and Resume ask
   * the same question before they touch the network.
   */
  function sharedRowTarget(row: FleetActiveWork) {
    const snapshot = liveActivity.hosts[row.hostId];
    const host = hosts.all.find((candidate) => candidate.id === row.hostId);
    if (
      !snapshot ||
      host?.status !== "ready" ||
      snapshot.stale ||
      snapshot.routeUrl !== row.routeUrl ||
      snapshot.instanceId !== row.instanceId ||
      host?.baseUrl !== row.routeUrl ||
      host.instanceId !== row.instanceId
    ) {
      toasts.push("This machine changed. Refresh its jobs and try again.", "error");
      void liveActivity.refresh();
      return null;
    }
    return snapshot.target;
  }

  /**
   * Whether this row is an auto-chained video the host parked at shutdown.
   *
   * A clip longer than the checkpoint can denoise in one pass runs as a chain
   * job. Graceful shutdown parks it as `paused` with its manifest, source
   * media, completed clips, and tail cache preserved precisely so it can be
   * resumed — the server answers `can_cancel: false` while parked, so Stop is
   * correctly absent and Resume is the only thing left to offer.
   */
  function canResumeChain(row: QueueRow): boolean {
    return (
      row.kind === "shared" &&
      connectedRow(row) &&
      row.shared.execution === "chain" &&
      row.shared.phase === "paused"
    );
  }

  /**
   * Resume a parked chain job on its own route. It lives in the chain-job id
   * space, so the generation queue's per-job pause has no row to act on and
   * `pauseEntries` stays empty for it.
   */
  async function resumeSharedChain(row: FleetActiveWork) {
    const target = sharedRowTarget(row);
    if (!target) return;
    await reserveMutation(
      { kind: "shared", key: row.key, createdAtMs: row.created_at_unix_ms, shared: row },
      async () => {
        try {
          const fresh = await apiJsonTo<{
            instance_id: string;
            items: { id: string; kind: string; execution: string | null; phase: string }[];
          }>(target, "/api/activity");
          const current = fresh.items.find((item) => item.id === row.id && item.kind === row.kind);
          if (
            fresh.instance_id !== row.instanceId ||
            current?.execution !== "chain" ||
            current.phase !== "paused" ||
            !sharedRowTarget(row)
          )
            throw new Error(
              "This job or machine changed. Reopen its details before resuming work.",
            );
          await apiFetchTo(target, `/api/chain-jobs/${encodeURIComponent(row.id)}/resume`, {
            method: "POST",
          });
          toasts.push("Resumed");
        } catch (error) {
          report(error);
        } finally {
          await liveActivity.refresh();
        }
      },
    );
  }

  async function cancelShared(row: FleetActiveWork) {
    if (cancellingShared.value.includes(row.key)) return;
    const target = sharedRowTarget(row);
    if (!target) return;
    cancellingShared.value = [...cancellingShared.value, row.key];
    try {
      const fresh = await apiJsonTo<{
        instance_id: string;
        items: { id: string; kind: string; phase: string; can_cancel: boolean }[];
      }>(target, "/api/activity");
      const current = fresh.items.find((item) => item.id === row.id && item.kind === row.kind);
      if (
        fresh.instance_id !== row.instanceId ||
        !current?.can_cancel ||
        !["queued", "paused", "held", "running", "preparing"].includes(current.phase) ||
        (row.phase === "held" && current.phase !== "held")
      )
        throw new Error("This job or machine changed. Reopen its details before stopping work.");
      if (
        row.execution !== "chain" &&
        ["running", "preparing"].includes(current.phase) &&
        queueFor(row.hostId)?.caps?.canCancelRunning !== true
      )
        throw new Error("This machine cannot stop a running job.");
      if (!sharedRowTarget(row)) return;
      if (row.execution === "chain") {
        await apiFetchTo(target, `/api/chain-jobs/${encodeURIComponent(row.id)}/cancel`, {
          method: "POST",
        });
      } else {
        await jobs.cancelJob(row.hostId, row.id, { onlyHeld: row.phase === "held" });
      }
      const cached = liveActivity.hosts[row.hostId]?.items.find(
        (item) => item.kind === row.kind && item.id === row.id,
      );
      if (cached) {
        cached.can_cancel = false;
        cached.phase = "cancelling";
      }
      toasts.push("Stopped");
    } catch (error) {
      report(error);
    } finally {
      try {
        await liveActivity.refresh();
      } finally {
        cancellingShared.value = cancellingShared.value.filter((key) => key !== row.key);
      }
    }
  }

  function reservationKey(row: QueueRow): string {
    return row.kind === "shared" ? row.shared.key : row.key;
  }
  async function reserveMutation(row: QueueRow, action: () => Promise<unknown>): Promise<void> {
    if (!connectedRow(row)) return;
    const key = reservationKey(row);
    cancellingShared.value = [...cancellingShared.value, key];
    try {
      await action();
    } finally {
      try {
        await liveActivity.refresh();
      } finally {
        cancellingShared.value = cancellingShared.value.filter((value) => value !== key);
      }
    }
  }

  function connectedRow(row: QueueRow): boolean {
    if (cancellingShared.value.includes(reservationKey(row))) return false;
    const ref = serverRef(row);
    if (ref && jobs.isJobBusy(ref.hostId, ref.id)) return false;
    if (
      row.kind === "shared"
        ? cancellingShared.value.includes(row.shared.key)
        : row.print.cancelling || row.print.retrying
    )
      return false;
    const host = hosts.all.find((host) => host.id === hostIdFor(row));
    if (!host || host.status !== "ready") return false;
    if (row.kind === "print") return true;
    const snapshot = liveActivity.hosts[row.shared.hostId];
    return (
      !row.shared.stale &&
      !!row.shared.instanceId &&
      snapshot?.stale === false &&
      snapshot.instanceId === row.shared.instanceId &&
      host.instanceId === row.shared.instanceId &&
      snapshot.routeUrl === row.shared.routeUrl &&
      host.baseUrl === row.shared.routeUrl
    );
  }

  async function retryShared(row: FleetActiveWork) {
    const key = row.key;
    if (cancellingShared.value.includes(key)) return;
    if (!sharedRowTarget(row)) return;
    cancellingShared.value = [...cancellingShared.value, key];
    try {
      const entry = await jobs.queueJob(row.hostId, row.id);
      if (!sharedRowTarget(row)) return;
      const target = sharedRowTarget(row);
      if (!target || !row.instanceId) return;
      if (await missingQueueModel(target, row.instanceId, row.id))
        await startQueueDownloadRecovery(target, row.instanceId, row.id, row.hostLabel);
      else await jobs.retryJob(row.hostId, entry);
    } catch (error) {
      report(error);
    } finally {
      try {
        await liveActivity.refresh();
      } finally {
        cancellingShared.value = cancellingShared.value.filter((value) => value !== key);
      }
    }
  }

  function canCancel(row: QueueRow): boolean {
    if (!connectedRow(row)) return false;
    if (row.kind === "print") {
      const executing = ["loading", "denoising", "finishing"].includes(row.print.status);
      if (
        row.print.id &&
        executing &&
        !row.print.chainStageCount &&
        queueFor(hostIdFor(row))?.caps?.canCancelRunning !== true
      )
        return false;
      return (
        row.print.status !== "complete" && row.print.status !== "error" && !row.print.cancelling
      );
    }
    if (row.kind === "shared") {
      return (
        row.shared.kind === "generation" &&
        row.shared.can_cancel &&
        ["queued", "paused", "held", "running", "preparing"].includes(row.shared.phase) &&
        (row.shared.execution === "chain" ||
          !["running", "preparing"].includes(row.shared.phase) ||
          queueFor(row.shared.hostId)?.caps?.canCancelRunning === true) &&
        !row.shared.stale &&
        !cancellingShared.value.includes(row.shared.key)
      );
    }
    return false;
  }

  async function cancel(row: QueueRow) {
    if (!canCancel(row)) return;
    if (row.kind === "print") {
      const ref = serverRef(row);
      if (ref) await jobs.withJobMutation(ref.hostId, ref.id, () => cancelPrint(row.print));
      else await cancelPrint(row.print);
    } else await cancelShared(row.shared);
  }

  /** The host a row's server queue lives on, and the row's server id. A long
   * clip the host chained answers with its chain job, which lives in a
   * different id space from the generation queue — so it resolves a HOST but
   * never matches a queue entry, and the reorder and per-job pause entries
   * stay empty for it. */
  function serverRef(row: QueueRow): { hostId: string; id: string } | null {
    if (row.kind === "print") {
      return row.print.id ? { hostId: row.print.hostId ?? "local", id: row.print.id } : null;
    }
    if (
      row.kind === "shared" &&
      row.shared.kind === "generation" &&
      row.shared.execution !== "chain"
    ) {
      return { hostId: row.shared.hostId, id: row.shared.id };
    }
    return null;
  }

  /** The machine a row's work runs on. Unlike `serverRef` this answers for a
   * print that has no server id yet, because the queue it belongs to is
   * already decided. */
  function hostIdFor(row: QueueRow): string | null {
    if (row.kind === "print") return row.print.hostId ?? hosts.primaryHost?.id ?? "local";
    return row.shared.hostId;
  }

  /** This row's slot among its host's QUEUED entries — the index space the
   * reorder PATCH uses. A queue position counts running jobs too, so nudging
   * against it is off-by-N the moment anything on the host is running. */
  function queuedIndexOf(row: QueueRow): number {
    const ref = serverRef(row);
    if (!ref) return -1;
    return (
      jobs.queues[ref.hostId]?.entries
        .filter((entry) => entry.state === "queued")
        .findIndex((entry) => entry.id === ref.id) ?? -1
    );
  }

  function canReorder(row: QueueRow): boolean {
    const ref = serverRef(row);
    return (
      connectedRow(row) &&
      ref !== null &&
      jobs.queues[ref.hostId]?.caps?.canReorder === true &&
      queuedIndexOf(row) >= 0
    );
  }

  /** Move a waiting row to `position` among its host's queued rows; the
   * server clamps and re-syncs, so an out-of-range index is harmless. */
  async function reorder(row: QueueRow, position: number) {
    const ref = serverRef(row);
    if (!ref || !canReorder(row)) return;
    await reserveMutation(row, () => jobs.reorderQueued(ref.hostId, ref.id, Math.max(0, position)));
  }

  /**
   * A drag lands the dragged row in the SLOT of the row it was dropped on.
   * Both rows must be queued entries of the same host: a position is an index
   * into one host's line, so a cross-machine drop would silently reorder a
   * queue the user was not pointing at.
   */
  async function dropOn(dragged: QueueRow, target: QueueRow) {
    const from = serverRef(dragged);
    const to = serverRef(target);
    if (!from || !to || from.hostId !== to.hostId || from.id === to.id) return;
    if (!canReorder(dragged) || !canReorder(target)) return;
    await reorder(dragged, queuedIndexOf(target)).catch(report);
  }

  /** This row's entry in its host's live queue listing, when it has one. */
  function queueEntryOf(row: QueueRow) {
    const ref = serverRef(row);
    if (!ref) return null;
    return jobs.queues[ref.hostId]?.entries.find((entry) => entry.id === ref.id) ?? null;
  }

  /**
   * Pause or resume ONE waiting row, where the host offers per-job pause.
   * Only queued and already-paused rows have an API: a running job cannot be
   * suspended, so the entry is absent rather than disabled.
   */
  function pauseEntries(row: QueueRow): MenuEntry[] {
    const ref = serverRef(row);
    if (!ref || !connectedRow(row) || jobs.queues[ref.hostId]?.caps?.canPauseJob !== true)
      return [];
    const entry = queueEntryOf(row);
    if (!entry || (entry.state !== "queued" && entry.state !== "paused")) return [];
    const paused = entry.state === "paused";
    return [
      {
        label: paused ? "Resume" : "Pause",
        action: () => {
          if (connectedRow(row))
            void reserveMutation(row, () => jobs.setJobPaused(ref.hostId, ref.id, !paused)).catch(
              report,
            );
        },
      },
    ];
  }

  /** The reorder entries for a waiting row, or nothing where the host does not offer it. */
  function reorderEntries(row: QueueRow): MenuEntry[] {
    if (!canReorder(row)) return [];
    const at = queuedIndexOf(row);
    return [
      { label: "Jump the line", disabled: at === 0, action: () => void reorder(row, 0) },
      { label: "Move earlier", disabled: at === 0, action: () => void reorder(row, at - 1) },
      { label: "Move later", action: () => void reorder(row, at + 1) },
      { separator: true },
    ];
  }

  /** A held print (a style still downloading, a machine that refused for
   * now) is retried on its own host through the store's fence. */
  function retry(job: Job) {
    if (!job.retryable || job.retrying) return;
    if (["MODEL_NOT_FOUND", "UNKNOWN_MODEL"].includes(job.holdCode ?? "")) {
      const host = hosts.all.find((host) => host.id === (job.hostId ?? "local"));
      if (host?.status === "ready" && host.baseUrl && host.instanceId && job.id)
        void startQueueDownloadRecovery(
          { baseUrl: host.baseUrl, apiKey: host.apiKey },
          host.instanceId,
          job.id,
          host.label,
        );
      return;
    }
    void generation
      .retryHeld(job.clientId)
      .then(() => toasts.push(`Retry queued on ${job.hostLabel ?? "this machine"}.`))
      .catch(report);
  }

  async function stopEverything() {
    await Promise.all(queue.rows.value.filter(canCancel).map((row) => cancel(row)));
    for (const host of hosts.all) {
      if (host.status === "ready" && jobs.queues[host.id]?.caps?.canCancelAll) {
        await jobs.cancelAll(host.id).catch(report);
      }
    }
  }

  /** Every machine the fan-out would actually change: one that owns a
   * cancellable row, or one whose queue offers Cancel all AND holds work. An
   * idle machine is sent the same Cancel all, but nothing on it is lost, so
   * the sentence must not count it. Never below one: the display host is
   * always the machine the question is about. */
  const stopEverythingHostCount = computed(() => {
    const ids = new Set<string>();
    for (const row of queue.rows.value) {
      if (!canCancel(row)) continue;
      const id = hostIdFor(row);
      if (id) ids.add(id);
    }
    for (const host of hosts.all) {
      const queued = jobs.queues[host.id];
      if (host.status !== "ready" || !queued?.caps?.canCancelAll) continue;
      if (queued.entries.length > 0) ids.add(host.id);
    }
    return Math.max(ids.size, 1);
  });

  const stopEverythingSummary = computed(() => {
    const pictures = queue.liveCount.value;
    const machines = stopEverythingHostCount.value;
    return (
      `Stops ${pictures} ${pictures === 1 ? "picture" : "pictures"} on ` +
      `${machines} ${machines === 1 ? "machine" : "machines"}. ` +
      "Anything part-finished is lost."
    );
  });

  function askStopEverything() {
    stopEverythingOpen.value = true;
  }

  function cancelStopEverything() {
    if (!stopEverythingBusy.value) stopEverythingOpen.value = false;
  }

  async function confirmStopEverything() {
    if (stopEverythingBusy.value) return;
    stopEverythingBusy.value = true;
    try {
      await stopEverything();
    } finally {
      stopEverythingBusy.value = false;
      stopEverythingOpen.value = false;
    }
  }

  function openPrint(job: Job) {
    generation.select(job.clientId);
    if (job.request) composer.set({ request: job.request });
    if (job.status === "complete") {
      if (job.result?.filename) {
        // The store no-ops on a fresh URL and re-mints an expired media
        // ticket, so a print re-selected an hour later opens on the canvas
        // instead of failing its fetch against a dead URL.
        void generation.refreshRemoteResultUrl(job.clientId).catch(() => {
          toasts.push("Open this older print in My images");
          void router.push("/library");
        });
      } else if (!job.resultUrl) {
        toasts.push("Open this older print in My images");
        void router.push("/library");
        return;
      }
    }
    void router.push("/create");
  }

  function open(row: QueueRow) {
    if (row.kind === "print") openPrint(row.print);
    else void openLiveWork(row.shared);
  }

  function menu(row: QueueRow): MenuEntry[] {
    const source = serverRef(row);
    const held =
      queueEntryOf(row)?.state === "held" || (row.kind === "print" && row.print.holdError !== null);
    const transferEntries: MenuEntry[] =
      held && connectedRow(row) && source && transfer?.canSend(source.hostId)
        ? [
            {
              label: "Send to another machine…",
              action: () => transfer.open(source.hostId, source.id),
            },
          ]
        : [];
    const sharedEntry = row.kind === "shared" ? queueEntryOf(row) : null;
    const sharedRetry =
      sharedEntry?.state === "held" &&
      sharedEntry.retryable !== false &&
      !!sharedEntry.batch_id?.trim() &&
      !!sharedEntry.client_batch_id?.trim();
    if (row.kind === "shared") {
      return [
        { label: "Show details", action: () => open(row) },
        ...transferEntries,
        ...(sharedRetry && sharedEntry
          ? [
              {
                label: "Retry now",
                disabled: !connectedRow(row) || cancellingShared.value.includes(row.shared.key),
                action: () => void retryShared(row.shared),
              },
            ]
          : []),
        ...reorderEntries(row),
        ...pauseEntries(row),
        ...(canResumeChain(row)
          ? [{ label: "Resume", action: () => void resumeSharedChain(row.shared) }]
          : []),
        { label: "Stop", danger: true, disabled: !canCancel(row), action: () => void cancel(row) },
      ];
    }
    const job = row.print;
    const live = job.status !== "complete" && job.status !== "error";
    return [
      ...transferEntries,
      ...reorderEntries(row),
      ...pauseEntries(row),
      live
        ? { label: "Stop", danger: true, disabled: !canCancel(row), action: () => void cancel(row) }
        : {
            label: "Remove from queue",
            disabled: !jobCanBeRemoved(job),
            action: () => generation.removeSettled(job.clientId),
          },
      ...(job.retryable
        ? [
            {
              label: ["MODEL_NOT_FOUND", "UNKNOWN_MODEL"].includes(job.holdCode ?? "")
                ? "Download and Retry"
                : "Retry now",
              disabled: !connectedRow(row),
              action: () => {
                if (connectedRow(row)) retry(job);
              },
            },
          ]
        : []),
      { separator: true },
      {
        label: "Use these words",
        action: () => {
          composer.set({
            prompt: job.prompt,
            model: job.model,
            seed: null,
            width: job.width,
            height: job.height,
            steps: job.total,
            guidance: job.guidance,
          });
          void router.push("/create");
        },
      },
      {
        label: "Show in My images",
        disabled: job.status !== "complete",
        action: () => void router.push("/library"),
      },
      { separator: true },
      {
        label: "Clear finished",
        disabled: !generation.jobs.some((j) => j.status === "complete" || j.status === "error"),
        action: () => generation.prune(0),
      },
    ];
  }

  function openContextMenu(event: MouseEvent, row: QueueRow) {
    contextMenu.open(event, menu(row));
  }

  return {
    canPause,
    paused,
    togglePause,
    canPauseFor,
    pausedFor,
    togglePauseFor,
    hostIdFor,
    stopEverything,
    askStopEverything,
    confirmStopEverything,
    cancelStopEverything,
    stopEverythingOpen,
    stopEverythingBusy,
    stopEverythingSummary,
    canCancel,
    cancel,
    canReorder,
    dropOn,
    open,
    contextMenu: openContextMenu,
    cancellingShared,
  };
}
