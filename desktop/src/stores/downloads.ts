import { defineStore } from "pinia";
import { markRaw } from "vue";
import {
  apiFetch,
  apiFetchTo,
  apiJson,
  ApiError,
  currentTarget,
  type ApiTarget,
} from "../lib/api/client";
import { sseStream } from "../lib/api/sse";
import { notifyPulled } from "../lib/notify";
import { useModelStore } from "./models";
import { useHostModelsStore } from "./hostModels";
import { useToastStore } from "./toasts";
import type { HostView } from "./hosts";
import type { DownloadEvent, DownloadJob, DownloadsListing } from "../lib/api/types";

export interface DownloadsState {
  activeJobs: DownloadJob[];
  queued: DownloadJob[];
  history: DownloadJob[];
}

export function emptyDownloadsState(): DownloadsState {
  return { activeJobs: [], queued: [], history: [] };
}

export interface DownloadHostState extends DownloadsState {
  label: string;
  target: ApiTarget;
  subscribed: boolean;
  abort: AbortController | null;
  cancelling: string[];
  ready: Promise<void> | null;
}

export interface HostedDownloadJob {
  hostId: string;
  hostLabel: string | null;
  job: DownloadJob;
}

const STREAM_OPEN_TIMEOUT_MS = 10_000;

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
  const job = state.activeJobs.find((j) => j.id === id) ?? state.queued.find((j) => j.id === id);
  const activeJobs = state.activeJobs.filter((j) => j.id !== id);
  const queued = state.queued.filter((j) => j.id !== id);
  const history = job
    ? [{ ...job, status, error: error ?? job.error ?? null }, ...state.history]
    : state.history;
  return { activeJobs, queued, history };
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
        activeJobs: ev.listing.active_jobs ?? (ev.listing.active ? [ev.listing.active] : []),
        queued: [...ev.listing.queued],
        history: [...ev.listing.history],
      };
    case "enqueued":
      if (state.queued.some((j) => j.id === ev.id) || state.activeJobs.some((j) => j.id === ev.id))
        return state;
      return { ...state, queued: [...state.queued, synthQueued(ev.id, ev.model)] };
    case "dequeued":
      return { ...state, queued: state.queued.filter((j) => j.id !== ev.id) };
    case "started": {
      const base =
        state.queued.find((j) => j.id === ev.id) ??
        state.activeJobs.find((j) => j.id === ev.id) ??
        synthQueued(ev.id, "");
      const active: DownloadJob = {
        ...base,
        status: "active",
        files_total: ev.files_total,
        bytes_total: ev.bytes_total,
      };
      return {
        activeJobs: [...state.activeJobs.filter((j) => j.id !== ev.id), active],
        queued: state.queued.filter((j) => j.id !== ev.id),
        history: state.history,
      };
    }
    case "progress":
      if (!state.activeJobs.some((job) => job.id === ev.id)) return state;
      return {
        ...state,
        activeJobs: state.activeJobs.map((job) =>
          job.id === ev.id
            ? {
                ...job,
                files_done: ev.files_done,
                bytes_done: ev.bytes_done,
                current_file: ev.current_file ?? job.current_file ?? null,
              }
            : job,
        ),
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
    cancelling: [] as string[],
    ready: null as Promise<void> | null,
    primaryTarget: null as ApiTarget | null,
    primaryHostId: "primary",
    hostStates: {} as Record<string, DownloadHostState>,
  }),
  getters: {
    /** In-flight rows for the tray: the active job first, then the queue. */
    inFlight(state): DownloadJob[] {
      return [...state.activeJobs, ...state.queued];
    },
    hostedInFlight(state): HostedDownloadJob[] {
      const primary = [...state.activeJobs, ...state.queued].map((job) => ({
        hostId: state.primaryHostId,
        hostLabel: null,
        job,
      }));
      const extra = Object.entries(state.hostStates).flatMap(([hostId, host]) =>
        [...host.activeJobs, ...host.queued].map((job) => ({
          hostId,
          hostLabel: host.label,
          job,
        })),
      );
      return [...primary, ...extra];
    },
    hasActivity(): boolean {
      return this.hostedInFlight.length > 0;
    },
    isCancelling(state) {
      return (hostId: string, id: string): boolean =>
        hostId === state.primaryHostId
          ? state.cancelling.includes(id)
          : (state.hostStates[hostId]?.cancelling.includes(id) ?? false);
    },
  },
  actions: {
    async fetch() {
      const listing = await apiJson<DownloadsListing>("/api/downloads");
      this.apply({ type: "snapshot", listing });
    },
    apply(ev: DownloadEvent) {
      const next = applyDownloadEvent(this.$state, ev);
      this.activeJobs = next.activeJobs;
      this.queued = next.queued;
      this.history = next.history;
      if (ev.type === "job_cancelled" || ev.type === "job_done" || ev.type === "job_failed") {
        this.cancelling = this.cancelling.filter((id) => id !== ev.id);
      }
    },
    /** Subscribe to a host's download stream. Idempotent for each host. */
    subscribe(host?: HostView): Promise<void> {
      if (host && !host.primary) {
        return this.subscribeHost(host);
      }
      const target = host?.baseUrl
        ? { baseUrl: host.baseUrl, apiKey: host.apiKey }
        : currentTarget();
      const hostId = host?.id ?? "primary";
      if (host) this.unsubscribeHost(host.id);
      if (
        this.subscribed &&
        this.primaryTarget?.baseUrl === target.baseUrl &&
        this.primaryTarget.apiKey === target.apiKey
      ) {
        this.primaryHostId = hostId;
        return this.ready ?? Promise.resolve();
      }
      if (this.subscribed) this.unsubscribe();
      this.activeJobs = [];
      this.queued = [];
      this.history = [];
      this.subscribed = true;
      this.primaryTarget = target;
      this.primaryHostId = hostId;
      const abort = markRaw(new AbortController());
      let markReady!: () => void;
      let failReady!: (error: Error) => void;
      let readyTimer: ReturnType<typeof setTimeout>;
      const ready = markRaw(
        new Promise<void>((resolve, reject) => {
          markReady = () => {
            clearTimeout(readyTimer);
            resolve();
          };
          failReady = (error) => {
            clearTimeout(readyTimer);
            abort.abort();
            if (this.abort === abort) this.unsubscribe();
            reject(error);
          };
          readyTimer = setTimeout(
            () => failReady(new Error(`Timed out connecting to ${host?.label ?? "download host"}`)),
            STREAM_OPEN_TIMEOUT_MS,
          );
        }),
      );
      void ready.catch(() => {});
      this.abort = abort;
      this.ready = ready;
      void sseStream("/api/downloads/stream", {
        target,
        signal: abort.signal,
        retry: true,
        onOpen: markReady,
        onOpenError: failReady,
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
      return ready;
    },
    subscribeHost(host: HostView): Promise<void> {
      if (!host.baseUrl) return Promise.reject(new Error(`${host.label} has no API endpoint`));
      const target = { baseUrl: host.baseUrl, apiKey: host.apiKey };
      const existing = this.hostStates[host.id];
      if (
        existing?.subscribed &&
        existing.target.baseUrl === target.baseUrl &&
        existing.target.apiKey === target.apiKey
      )
        return existing.ready ?? Promise.resolve();
      if (existing) this.unsubscribeHost(host.id);
      const abort = markRaw(new AbortController());
      let markReady!: () => void;
      let failReady!: (error: Error) => void;
      let readyTimer: ReturnType<typeof setTimeout>;
      const ready = markRaw(
        new Promise<void>((resolve, reject) => {
          markReady = () => {
            clearTimeout(readyTimer);
            resolve();
          };
          failReady = (error) => {
            clearTimeout(readyTimer);
            abort.abort();
            if (this.hostStates[host.id]?.abort === abort) this.unsubscribeHost(host.id);
            reject(error);
          };
          readyTimer = setTimeout(
            () => failReady(new Error(`Timed out connecting to ${host.label}`)),
            STREAM_OPEN_TIMEOUT_MS,
          );
        }),
      );
      void ready.catch(() => {});
      this.hostStates[host.id] = {
        ...(this.hostStates[host.id] ?? emptyDownloadsState()),
        label: host.label,
        target,
        subscribed: true,
        abort,
        cancelling: this.hostStates[host.id]?.cancelling ?? [],
        ready,
      };
      void sseStream("/api/downloads/stream", {
        target,
        signal: abort.signal,
        retry: true,
        onOpen: markReady,
        onOpenError: failReady,
        onEvent: (_event, data) => {
          try {
            const ev = JSON.parse(data) as DownloadEvent;
            this.applyForHost(host.id, ev);
            if (ev.type === "job_done") this.onJobComplete(ev.model, host.id);
          } catch {
            /* skip malformed frame */
          }
        },
      });
      return ready;
    },
    applyForHost(hostId: string, ev: DownloadEvent) {
      const host = this.hostStates[hostId];
      if (!host) return;
      const next = applyDownloadEvent(host, ev);
      host.activeJobs = next.activeJobs;
      host.queued = next.queued;
      host.history = next.history;
      if (ev.type === "job_cancelled" || ev.type === "job_done" || ev.type === "job_failed") {
        host.cancelling = host.cancelling.filter((id) => id !== ev.id);
      }
    },
    unsubscribe() {
      this.abort?.abort();
      this.abort = null;
      this.subscribed = false;
      this.ready = null;
      this.primaryTarget = null;
    },
    unsubscribeHost(hostId: string) {
      this.hostStates[hostId]?.abort?.abort();
      delete this.hostStates[hostId];
    },
    onJobComplete(model: string, hostId?: string) {
      const host = hostId ? this.hostStates[hostId] : null;
      useToastStore().push(`Pulled ${model}${host ? ` on ${host.label}` : ""}`);
      notifyPulled(model);
      if (host) void useHostModelsStore().refresh(true);
      else void useModelStore().fetch();
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
    async cancel(id: string, hostId?: string) {
      const resolvedHostId = hostId ?? this.primaryHostId;
      const isPrimary = resolvedHostId === this.primaryHostId;
      const host = isPrimary ? null : this.hostStates[resolvedHostId];
      const cancelling = host?.cancelling ?? this.cancelling;
      if (cancelling.includes(id)) return;
      cancelling.push(id);
      try {
        const path = `/api/downloads/${encodeURIComponent(id)}`;
        if (host) await apiFetchTo(host.target, path, { method: "DELETE" });
        else if (this.primaryTarget)
          await apiFetchTo(this.primaryTarget, path, { method: "DELETE" });
        else await apiFetch(path, { method: "DELETE" });
      } catch (error) {
        if (host) host.cancelling = host.cancelling.filter((candidate) => candidate !== id);
        else this.cancelling = this.cancelling.filter((candidate) => candidate !== id);
        throw error;
      }
    },
  },
});
