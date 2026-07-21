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
import { startCatalogDownload } from "../lib/api/catalog";
import { applyDownloadEvent, emptyDownloadsState, type DownloadsState } from "../lib/downloads";
import { notifyPulled, notifyPullFailed } from "../lib/notify";
import { useModelStore } from "./models";
import { useHostModelsStore } from "./hostModels";
import { useToastStore } from "./toasts";
import type { HostView } from "./hosts";
import type { DownloadEvent, DownloadJob, DownloadsListing } from "../lib/api/types";

export { applyDownloadEvent, emptyDownloadsState, type DownloadsState } from "../lib/downloads";

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

/** Scope prefix for the primary stream's rate samples (host streams use their id). */
const PRIMARY_RATE_SCOPE = "primary";

/** One byte-count observation on an active download. */
export interface RateSample {
  ts: number;
  bytes: number;
}

/** Sliding window width for download-rate sampling. */
export const RATE_WINDOW_MS = 10_000;

/**
 * Append a sample and trim entries older than the window. Mutates (and
 * returns) `samples` in place so reactive arrays keep their identity.
 */
export function pushRateSample(
  samples: RateSample[],
  bytes: number,
  now: number = Date.now(),
): RateSample[] {
  samples.push({ ts: now, bytes });
  while (samples.length > 0 && now - samples[0]!.ts > RATE_WINDOW_MS) samples.shift();
  return samples;
}

/**
 * Client-side ETA math — the server only emits raw byte counters. Returns
 * whole seconds remaining, or null when the window is too small, stalled,
 * or non-advancing (no rate → no honest ETA).
 */
export function computeEtaSeconds(samples: RateSample[], bytesTotal: number): number | null {
  if (samples.length < 2) return null;
  const first = samples[0]!;
  const last = samples[samples.length - 1]!;
  const deltaBytes = last.bytes - first.bytes;
  const deltaMs = last.ts - first.ts;
  if (deltaMs <= 0 || deltaBytes <= 0) return null;
  const ratePerSec = (deltaBytes * 1000) / deltaMs;
  const eta = Math.max(0, bytesTotal - last.bytes) / ratePerSec;
  return Number.isFinite(eta) ? Math.round(eta) : null;
}

/**
 * Newest-first by completion time. Entries without a `completed_at` keep their
 * incoming (server) order — `Array.prototype.sort` is stable, so equal keys
 * never reshuffle.
 */
function byCompletedDesc(a: DownloadJob, b: DownloadJob): number {
  return (b.completed_at ?? 0) - (a.completed_at ?? 0);
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
    /** Sliding rate windows keyed `<scope>:<job id>` (scope = "primary" or host id). */
    rateSamples: {} as Record<string, RateSample[]>,
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
    /** Settled rows for the tray's history section, newest first per host. */
    hostedHistory(state): HostedDownloadJob[] {
      const primary = [...state.history].sort(byCompletedDesc).map((job) => ({
        hostId: state.primaryHostId,
        hostLabel: null,
        job,
      }));
      const extra = Object.entries(state.hostStates).flatMap(([hostId, host]) =>
        [...host.history]
          .sort(byCompletedDesc)
          .map((job) => ({ hostId, hostLabel: host.label, job })),
      );
      return [...primary, ...extra];
    },
    /** Live ETA (seconds) per active job id, derived from the rate windows. */
    etaByJob(state): Record<string, number | null> {
      const out: Record<string, number | null> = {};
      for (const job of state.activeJobs) {
        out[job.id] = computeEtaSeconds(
          state.rateSamples[`${PRIMARY_RATE_SCOPE}:${job.id}`] ?? [],
          job.bytes_total,
        );
      }
      for (const [hostId, host] of Object.entries(state.hostStates)) {
        for (const job of host.activeJobs) {
          out[job.id] = computeEtaSeconds(
            state.rateSamples[`${hostId}:${job.id}`] ?? [],
            job.bytes_total,
          );
        }
      }
      return out;
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
      this.trackRates(PRIMARY_RATE_SCOPE, ev, next);
      if (ev.type === "job_cancelled" || ev.type === "job_done" || ev.type === "job_failed") {
        this.cancelling = this.cancelling.filter((id) => id !== ev.id);
      }
    },
    /**
     * Maintain the sliding rate windows behind `etaByJob`. Windows are
     * scoped per stream so one host's snapshot can't evict another's, and
     * settled/vanished jobs drop theirs so the map can't grow unbounded.
     */
    trackRates(scope: string, ev: DownloadEvent, next: DownloadsState) {
      switch (ev.type) {
        case "progress": {
          if (!next.activeJobs.some((job) => job.id === ev.id)) return;
          const key = `${scope}:${ev.id}`;
          const samples = this.rateSamples[key] ?? [];
          pushRateSample(samples, ev.bytes_done);
          this.rateSamples[key] = samples;
          return;
        }
        case "job_done":
        case "job_failed":
        case "job_cancelled":
          delete this.rateSamples[`${scope}:${ev.id}`];
          return;
        case "snapshot": {
          const live = new Set(
            [...next.activeJobs, ...next.queued].map((job) => `${scope}:${job.id}`),
          );
          for (const key of Object.keys(this.rateSamples)) {
            if (key.startsWith(`${scope}:`) && !live.has(key)) delete this.rateSamples[key];
          }
          return;
        }
        default:
          return;
      }
    },
    /** Drop every rate window belonging to a stream scope. */
    clearRates(scope: string) {
      for (const key of Object.keys(this.rateSamples)) {
        if (key.startsWith(`${scope}:`)) delete this.rateSamples[key];
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
      this.clearRates(PRIMARY_RATE_SCOPE);
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
            else if (ev.type === "job_failed") this.onJobFailed(ev.id);
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
        ...(existing ?? emptyDownloadsState()),
        label: host.label,
        target,
        subscribed: true,
        abort,
        cancelling: existing?.cancelling ?? [],
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
            else if (ev.type === "job_failed") this.onJobFailed(ev.id, host.id);
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
      this.trackRates(hostId, ev, next);
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
      this.clearRates(hostId);
    },
    onJobComplete(model: string, hostId?: string) {
      const host = hostId ? this.hostStates[hostId] : null;
      useToastStore().push(`Pulled ${model}${host ? ` on ${host.label}` : ""}`);
      notifyPulled(model);
      if (host) void useHostModelsStore().refresh(true);
      else void useModelStore().fetch();
    },
    /**
     * A pull failed on some host (§08 G11). `job_failed` carries no model, so
     * recover it from the just-settled history row, then toast + notify (native
     * only fires while backgrounded).
     */
    onJobFailed(id: string, hostId?: string) {
      const host = hostId ? this.hostStates[hostId] : null;
      const job = (host?.history ?? this.history).find((j) => j.id === id);
      const model = job?.model ?? "the model";
      const suffix = host ? ` on ${host.label}` : "";
      const detail = job?.error ? ` — ${job.error}` : "";
      useToastStore().push(`Couldn't pull ${model}${suffix}${detail}`, "error");
      notifyPullFailed(model, job?.error ?? undefined);
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
    /**
     * Re-start a settled (failed) job's download on the SAME host it ran on.
     * Reuses the catalog dispatcher path the Catalog view pulls through, so
     * catalog ids re-enqueue their companions and plain names hit
     * `/api/downloads`; credentials forward only to explicitly remote hosts.
     */
    async retry(hostId: string, job: DownloadJob) {
      const isPrimary = hostId === this.primaryHostId;
      const host = isPrimary ? null : this.hostStates[hostId];
      if (!isPrimary && !host) {
        throw new Error("That host is no longer connected.");
      }
      const target = host?.target ?? this.primaryTarget ?? currentTarget();
      try {
        await startCatalogDownload(job.model, target, !isPrimary);
      } catch (err) {
        if (err instanceof ApiError && err.status === 409) {
          useToastStore().push(`${job.model} is already queued.`);
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
