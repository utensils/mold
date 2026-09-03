import { computed, ref, type ComputedRef, type Ref } from "vue";
import { useRouter } from "vue-router";
import { apiFetchTo } from "@studio/api/client";
import type { FleetActiveWork } from "@studio/api/activity";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { useOpenLiveWork } from "./useOpenLiveWork";
import { useQueueActivity, type QueueRow } from "./useQueueActivity";
import { useChainJobsStore } from "../stores/chainJobs";
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
  /** Cancel every live print this client owns and every host's queue. */
  stopEverything(): Promise<void>;
  canCancel(row: QueueRow): boolean;
  cancel(row: QueueRow): Promise<void>;
  /** Bring the row to the canvas (or the surface that can inspect it). */
  open(row: QueueRow): void;
  contextMenu(event: MouseEvent, row: QueueRow): void;
  menu(row: QueueRow): MenuEntry[];
  cancellingShared: Ref<string[]>;
}

/**
 * Every action the queue offers, shared by the sidebar rail and the Queue
 * view so both surfaces act on the same authorities.
 */
export function useQueueCommands(): QueueCommands {
  const router = useRouter();
  const composer = useComposerStore();
  const contextMenu = useContextMenuStore();
  const draft = useSequenceDraftStore();
  const generation = useGenerationStore();
  const chainJobs = useChainJobsStore();
  const hosts = useHostsStore();
  const hostStatus = useHostStatusStore();
  const jobs = useJobsStore();
  const liveActivity = useLiveActivityStore();
  const toasts = useToastStore();
  const queue = useQueueActivity();
  const openLiveWork = useOpenLiveWork();
  const cancellingShared = ref<string[]>([]);

  const displayQueue = computed(() => {
    const host = hostStatus.displayHost;
    return host ? (jobs.queues[host.id] ?? null) : null;
  });
  const canPause = computed(() => displayQueue.value?.caps?.canPause === true);
  const paused = computed(() => displayQueue.value?.paused === true);

  function report(error: unknown) {
    toasts.push(error instanceof Error ? error.message : String(error), "error");
  }

  async function togglePause() {
    const host = hostStatus.displayHost;
    if (!host) return;
    try {
      if (paused.value) await jobs.resume(host.id);
      else await jobs.pause(host.id);
    } catch (error) {
      report(error);
    }
  }

  async function cancelPrint(job: Job) {
    try {
      if (await generation.cancel(job.clientId)) toasts.push("Stopped");
    } catch (error) {
      report(error);
    }
  }

  async function cancelShared(row: FleetActiveWork) {
    if (cancellingShared.value.includes(row.key)) return;
    const snapshot = liveActivity.hosts[row.hostId];
    const host = hosts.all.find((candidate) => candidate.id === row.hostId);
    if (
      !snapshot ||
      snapshot.stale ||
      snapshot.routeUrl !== row.routeUrl ||
      snapshot.instanceId !== row.instanceId ||
      host?.baseUrl !== row.routeUrl ||
      host.instanceId !== row.instanceId
    ) {
      toasts.push("This machine changed. Refresh its jobs and try again.", "error");
      void liveActivity.refresh();
      return;
    }
    cancellingShared.value = [...cancellingShared.value, row.key];
    try {
      if (row.execution === "chain") {
        await apiFetchTo(snapshot.target, `/api/chain-jobs/${encodeURIComponent(row.id)}/cancel`, {
          method: "POST",
        });
      } else {
        await jobs.cancelJob(row.hostId, row.id);
      }
      const current = liveActivity.hosts[row.hostId]?.items.find(
        (item) => item.kind === row.kind && item.id === row.id,
      );
      if (current) {
        current.can_cancel = false;
        current.phase = "cancelling";
      }
      toasts.push("Stopped");
    } catch (error) {
      report(error);
    } finally {
      await liveActivity.refresh();
      cancellingShared.value = cancellingShared.value.filter((key) => key !== row.key);
    }
  }

  function canCancel(row: QueueRow): boolean {
    if (row.kind === "sequence") return row.sequence.actions.includes("cancel");
    if (row.kind === "print") {
      return (
        row.print.status !== "complete" && row.print.status !== "error" && !row.print.cancelling
      );
    }
    if (row.kind === "shared") {
      return (
        row.shared.kind === "generation" &&
        row.shared.can_cancel &&
        !row.shared.stale &&
        !cancellingShared.value.includes(row.shared.key)
      );
    }
    return false;
  }

  async function cancel(row: QueueRow) {
    if (row.kind === "print") await cancelPrint(row.print);
    else if (row.kind === "shared") await cancelShared(row.shared);
    else await chainJobs.cancel(row.sequence.hostId, row.sequence.jobId).catch(report);
  }

  /** The host a row's server queue lives on, and the row's server id. */
  function serverRef(row: QueueRow): { hostId: string; id: string } | null {
    if (row.kind === "print") {
      return row.print.id ? { hostId: row.print.hostId ?? "local", id: row.print.id } : null;
    }
    if (row.kind === "shared" && row.shared.kind === "generation") {
      return { hostId: row.shared.hostId, id: row.shared.id };
    }
    return null;
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
      ref !== null && jobs.queues[ref.hostId]?.caps?.canReorder === true && queuedIndexOf(row) >= 0
    );
  }

  /** Move a waiting row to `position` among its host's queued rows; the
   * server clamps and re-syncs, so an out-of-range index is harmless. */
  async function reorder(row: QueueRow, position: number) {
    const ref = serverRef(row);
    if (!ref) return;
    await jobs.reorderQueued(ref.hostId, ref.id, Math.max(0, position));
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

  function openPrint(job: Job) {
    generation.select(job.clientId);
    draft.stopEditing();
    draft.output = "single";
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
    else if (row.kind === "sequence") {
      composer.setSequence({
        kind: "inspect",
        hostId: row.sequence.hostId,
        jobId: row.sequence.jobId,
      });
      void router.push("/create");
    } else void openLiveWork(row.shared);
  }

  function menu(row: QueueRow): MenuEntry[] {
    if (row.kind === "shared") {
      return [
        ...reorderEntries(row),
        { label: "Stop", danger: true, disabled: !canCancel(row), action: () => void cancel(row) },
      ];
    }
    if (row.kind === "sequence") {
      return [
        { label: "Open", action: () => open(row) },
        { label: "Stop", danger: true, disabled: !canCancel(row), action: () => void cancel(row) },
      ];
    }
    const job = row.print;
    const live = job.status !== "complete" && job.status !== "error";
    return [
      ...reorderEntries(row),
      live
        ? { label: "Stop", danger: true, disabled: !canCancel(row), action: () => void cancel(row) }
        : {
            label: "Remove from queue",
            disabled: !jobCanBeRemoved(job),
            action: () => generation.removeSettled(job.clientId),
          },
      ...(job.retryable
        ? [{ label: "Retry now", disabled: job.retrying, action: () => retry(job) }]
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
    stopEverything,
    canCancel,
    cancel,
    open,
    contextMenu: openContextMenu,
    menu,
    cancellingShared,
  };
}
