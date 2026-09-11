import type { ApiTarget } from "../api/client";
import type { FleetActiveWork } from "../api/activity";
import { findQueueEntryById } from "../api/queuePlan";
import {
  selectedQueueGeneration,
  type SelectedQueueGeneration,
} from "../api/generationSelection";
import { meshWorkflowRouteFor } from "../lib/meshWorkflowProvenance";

/** Somewhere a surface can navigate to: a path, or a path with a query. */
export type LiveWorkDestination =
  string | { path: string; query: Record<string, string> };

/** What opening live work needs from the surface doing the opening. */
export interface LiveWorkSurface<M extends object> {
  /** That machine's API target, or null when it is no longer connected. */
  targetFor: (hostId: string) => ApiTarget | null;
  /** Navigate. */
  go: (to: LiveWorkDestination) => Promise<void>;
  /** Say what went wrong, in this surface's own notification vocabulary. */
  fail: (message: string) => void;
  /** Hand restored settings to the surface that makes new pictures. */
  restore: (selection: SelectedQueueGeneration<M>, hostId: string) => void;
  /**
   * Where a server-owned chain row goes. The browser shows its machine; the
   * app shows the queue. Neither can re-enter the work itself.
   */
  chainDestination: (row: FleetActiveWork) => LiveWorkDestination;
  /**
   * What a download row opens. The browser raises its own downloads popover
   * and navigates nowhere, which is what a null answer means.
   */
  openDownloads: () => LiveWorkDestination | null;
}

/** The message a surface says when the machine has gone. */
export const LIVE_WORK_HOST_GONE = "That machine is no longer connected.";
/** The message a surface says when the host cannot hand the settings back. */
export const LIVE_WORK_NO_SETTINGS =
  "This machine cannot restore settings for that generation.";

/**
 * Opens server-owned work in whichever surface can inspect or resume it.
 *
 * A server-side chain row (a long video the host is auto-chaining, or a
 * `mold run --script` job) has no authoring surface any more: it goes where
 * the surface says rather than pretending New image can reattach to it. That
 * guard has to come FIRST, because such a row is `kind: "generation"` carrying
 * `execution: "chain"` — falling into the generation arm below would search
 * `/api/queue` for an id that only exists under `/api/chain-jobs` and dead-end
 * on "cannot restore settings".
 */
export function openLiveWorkWith<M extends object>(
  surface: LiveWorkSurface<M>,
): (row: FleetActiveWork) => Promise<void> {
  return async (row: FleetActiveWork) => {
    if (row.kind === "sequence" || row.execution === "chain") {
      await surface.go(surface.chainDestination(row));
      return;
    }
    if (row.kind === "generation") {
      const target = surface.targetFor(row.hostId);
      if (!target) {
        surface.fail(LIVE_WORK_HOST_GONE);
        return;
      }
      try {
        const entry = await findQueueEntryById(target, row.id);
        const selection = selectedQueueGeneration<M>(
          entry ? [entry] : [],
          row.id,
        );
        if (!selection) {
          surface.fail(LIVE_WORK_NO_SETTINGS);
          return;
        }
        // A 3-D Studio stage is admitted as an ordinary generation, so it
        // arrives here looking like any other print. New image cannot resume
        // a durable workflow — its stages, Cancel, Resume and history live
        // only under /api/mesh-workflows.
        const workflow = meshWorkflowRouteFor(selection.metadata, row.hostId);
        if (workflow) {
          await surface.go(workflow);
          return;
        }
        surface.restore(selection, row.hostId);
        await surface.go("/create");
      } catch (error) {
        surface.fail(error instanceof Error ? error.message : String(error));
      }
      return;
    }
    if (row.kind === "download") {
      const downloads = surface.openDownloads();
      if (downloads) await surface.go(downloads);
      return;
    }
    await surface.go(`/machines/${row.hostId}`);
  };
}
