import { defineStore } from "pinia";
import { apiFetch, apiJson, ApiError } from "../lib/api/client";
import { sseStream } from "../lib/api/sse";
import { useModelStore } from "./models";
import { useToastStore } from "./toasts";
import type { DownloadEvent, DownloadJob, DownloadsListing } from "../lib/api/types";

export interface DownloadsState {
  active: DownloadJob | null;
  queued: DownloadJob[];
  history: DownloadJob[];
}

export function emptyDownloadsState(): DownloadsState {
  return { active: null, queued: [], history: [] };
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
  const job = state.active?.id === id ? state.active : state.queued.find((j) => j.id === id);
  const active = state.active?.id === id ? null : state.active;
  const queued = state.queued.filter((j) => j.id !== id);
  const history = job
    ? [{ ...job, status, error: error ?? job.error ?? null }, ...state.history]
    : state.history;
  return { active, queued, history };
}

/**
 * Pure download-event reducer. The first frame is always a `snapshot` that
 * replaces state wholesale; subsequent deltas move a job queued → active →
 * history. Exported for tests.
 */
export function applyDownloadEvent(state: DownloadsState, ev: DownloadEvent): DownloadsState {
  switch (ev.type) {
    case "snapshot":
      return {
        active: ev.listing.active ?? null,
        queued: [...ev.listing.queued],
        history: [...ev.listing.history],
      };
    case "enqueued":
      if (state.queued.some((j) => j.id === ev.id) || state.active?.id === ev.id) return state;
      return { ...state, queued: [...state.queued, synthQueued(ev.id, ev.model)] };
    case "dequeued":
      return { ...state, queued: state.queued.filter((j) => j.id !== ev.id) };
    case "started": {
      const base =
        state.queued.find((j) => j.id === ev.id) ??
        (state.active?.id === ev.id ? state.active : synthQueued(ev.id, ""));
      const active: DownloadJob = {
        ...base,
        status: "active",
        files_total: ev.files_total,
        bytes_total: ev.bytes_total,
      };
      return { active, queued: state.queued.filter((j) => j.id !== ev.id), history: state.history };
    }
    case "progress":
      if (state.active?.id !== ev.id) return state;
      return {
        ...state,
        active: {
          ...state.active,
          files_done: ev.files_done,
          bytes_done: ev.bytes_done,
          current_file: ev.current_file ?? state.active.current_file ?? null,
        },
      };
    case "file_done":
      return state;
    case "job_done":
      return finishJob(state, ev.id, "completed");
    case "job_failed":
      return finishJob(state, ev.id, "failed", ev.error);
    case "job_cancelled":
      return finishJob(state, ev.id, "cancelled");
    case "catalog_ready":
      return state;
    default:
      return state;
  }
}

export const useDownloadsStore = defineStore("downloads", {
  state: () => ({
    ...emptyDownloadsState(),
    subscribed: false,
    abort: null as AbortController | null,
  }),
  getters: {
    /** In-flight rows for the tray: the active job first, then the queue. */
    inFlight(state): DownloadJob[] {
      return state.active ? [state.active, ...state.queued] : state.queued;
    },
    hasActivity(): boolean {
      return this.inFlight.length > 0;
    },
  },
  actions: {
    async fetch() {
      const listing = await apiJson<DownloadsListing>("/api/downloads");
      this.apply({ type: "snapshot", listing });
    },
    apply(ev: DownloadEvent) {
      const next = applyDownloadEvent(this.$state, ev);
      this.active = next.active;
      this.queued = next.queued;
      this.history = next.history;
    },
    /** Subscribe to the download stream. Idempotent — safe to call on mount. */
    subscribe() {
      if (this.subscribed) return;
      this.subscribed = true;
      const abort = new AbortController();
      this.abort = abort;
      void sseStream("/api/downloads/stream", {
        signal: abort.signal,
        retry: true,
        onEvent: (_event, data) => {
          try {
            const ev = JSON.parse(data) as DownloadEvent;
            this.apply(ev);
            if (ev.type === "job_done") this.onJobComplete(ev.model);
          } catch {
            /* skip malformed frame */
          }
        },
      });
    },
    unsubscribe() {
      this.abort?.abort();
      this.abort = null;
      this.subscribed = false;
    },
    onJobComplete(model: string) {
      useToastStore().push(`Pulled ${model}`);
      void useModelStore().fetch();
    },
    /** Enqueue a plain-name model. A 409 means it's already queued/installed. */
    async createDownload(model: string) {
      const toasts = useToastStore();
      try {
        await apiFetch("/api/downloads", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model }),
        });
      } catch (err) {
        if (err instanceof ApiError && err.status === 409) {
          toasts.push(`${model} is already queued.`);
          return;
        }
        throw err;
      }
    },
    async cancel(id: string) {
      await apiFetch(`/api/downloads/${encodeURIComponent(id)}`, { method: "DELETE" });
      this.apply({ type: "job_cancelled", id });
    },
  },
});
