import type { DownloadEvent, DownloadJob } from "./api/types";

export interface DownloadsState {
  activeJobs: DownloadJob[];
  queued: DownloadJob[];
  history: DownloadJob[];
}

export function emptyDownloadsState(): DownloadsState {
  return { activeJobs: [], queued: [], history: [] };
}

function synthQueued(id: string, model: string): DownloadJob {
  return {
    id,
    model,
    status: "queued",
    files_done: 0,
    files_total: 0,
    bytes_done: 0,
    bytes_total: 0,
  };
}

/** Move a settling job out of active/queued into history with a final status. */
function finishJob(
  state: DownloadsState,
  id: string,
  status: DownloadJob["status"],
  error?: string,
): DownloadsState {
  const job =
    state.activeJobs.find((candidate) => candidate.id === id) ??
    state.queued.find((candidate) => candidate.id === id);
  const activeJobs = state.activeJobs.filter((candidate) => candidate.id !== id);
  const queued = state.queued.filter((candidate) => candidate.id !== id);
  const history = job
    ? [{ ...job, status, error: error ?? job.error ?? null }, ...state.history]
    : state.history;
  return { activeJobs, queued, history };
}

/**
 * Pure download-event reducer. The first frame is always a `snapshot` that
 * replaces state wholesale; subsequent deltas move a job queued → active →
 * history. Shared by the desktop and remote-only mobile authorities.
 */
export function applyDownloadEvent(state: DownloadsState, event: DownloadEvent): DownloadsState {
  switch (event.type) {
    case "snapshot":
      return {
        activeJobs:
          event.listing.active_jobs ?? (event.listing.active ? [event.listing.active] : []),
        queued: [...event.listing.queued],
        history: [...event.listing.history],
      };
    case "enqueued":
      if (
        state.queued.some((job) => job.id === event.id) ||
        state.activeJobs.some((job) => job.id === event.id)
      ) {
        return state;
      }
      return {
        ...state,
        queued: [...state.queued, synthQueued(event.id, event.model)],
      };
    case "dequeued":
      return { ...state, queued: state.queued.filter((job) => job.id !== event.id) };
    case "started": {
      const activeIndex = state.activeJobs.findIndex((job) => job.id === event.id);
      const base =
        state.queued.find((job) => job.id === event.id) ??
        state.activeJobs[activeIndex] ??
        synthQueued(event.id, "");
      const active: DownloadJob = {
        ...base,
        status: "active",
        files_total: event.files_total,
        bytes_total: event.bytes_total,
      };
      const activeJobs = [...state.activeJobs];
      if (activeIndex >= 0) activeJobs.splice(activeIndex, 1, active);
      else activeJobs.push(active);
      return {
        activeJobs,
        queued: state.queued.filter((job) => job.id !== event.id),
        history: state.history,
      };
    }
    case "progress":
      if (!state.activeJobs.some((job) => job.id === event.id)) return state;
      return {
        ...state,
        activeJobs: state.activeJobs.map((job) =>
          job.id === event.id
            ? {
                ...job,
                files_done: event.files_done,
                bytes_done: event.bytes_done,
                current_file: event.current_file ?? job.current_file ?? null,
              }
            : job,
        ),
      };
    case "file_done":
      return {
        ...state,
        activeJobs: state.activeJobs.map((job) =>
          job.id === event.id
            ? { ...job, files_done: Math.min(job.files_total, job.files_done + 1) }
            : job,
        ),
      };
    case "job_done":
      return finishJob(state, event.id, "completed");
    case "job_failed":
      return finishJob(state, event.id, "failed", event.error);
    case "job_cancelled":
      return finishJob(state, event.id, "cancelled");
    case "catalog_ready":
      return state;
    default:
      return state;
  }
}
