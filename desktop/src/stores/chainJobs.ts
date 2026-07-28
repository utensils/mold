import { defineStore } from "pinia";
import {
  applyChainJobEvent,
  emptyChainJobLive,
  type ChainJobLive,
} from "@studio/lib/chainJobEvents";
import type {
  AmendRequest,
  AmendResponse,
  ChainJobDetail,
  ChainJobEvent,
  ChainJobListing,
  ChainJobSummary,
  ChainRequestWire,
  CreateChainJobResponse,
} from "@studio/lib/api/chainTypes";
import { apiFetchTo, apiJsonTo, type ApiTarget } from "../lib/api/client";
import { sseStream } from "../lib/api/sse";
import { notifyChainFinished } from "../lib/notify";
import type { GcOutcome, RetakeRequest } from "../lib/api/types";
import { useHostsStore } from "./hosts";
import { useToastStore } from "./toasts";

/** One host's durable chain-job listing. */
export interface HostChainJobs {
  jobs: ChainJobSummary[];
  error: string | null;
}

const POLL_INTERVAL_MS = 10_000;

const isActive = (job: ChainJobSummary) => job.state === "running" || job.state === "queued";

const jsonInit = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

/**
 * Durable sequence (chain) jobs across EVERY connected host. Each action
 * addresses a host by id and resolves its `ApiTarget` from the hosts store
 * at call time — the previous store kept one mutable `target`, so cancel /
 * remove / retake / gc silently hit whichever host was fetched last.
 */
export const useChainJobsStore = defineStore("chainJobs", {
  state: () => ({
    byHost: {} as Record<string, HostChainJobs>,
    /** The one job whose SSE stream feeds `live`; null when idle. */
    watching: null as { hostId: string; jobId: string } | null,
    live: emptyChainJobLive() as ChainJobLive,
    abort: null as AbortController | null,
    pollTimer: null as ReturnType<typeof setInterval> | null,
    /** Bumped whenever the watched job emits a `finalized` frame — the
     *  Create canvas watches it to refresh the gallery. */
    finalizedTick: 0,
    fetchVersions: {} as Record<string, number>,
  }),
  getters: {
    /** Every host's jobs flattened for the activity strip, newest first. */
    allJobs(state): { hostId: string; job: ChainJobSummary }[] {
      return Object.entries(state.byHost)
        .flatMap(([hostId, entry]) => entry.jobs.map((job) => ({ hostId, job })))
        .sort((a, b) => b.job.created_at_unix_ms - a.job.created_at_unix_ms);
    },
    anyActive(state): boolean {
      return Object.values(state.byHost).some((entry) => entry.jobs.some(isActive));
    },
  },
  actions: {
    /** Resolve a host id to its live ApiTarget — at CALL time, never cached. */
    targetFor(hostId: string): ApiTarget {
      const host = useHostsStore().all.find((h) => h.id === hostId);
      if (!host?.baseUrl) throw new Error(`Host ${hostId} is not connected.`);
      return { baseUrl: host.baseUrl, apiKey: host.apiKey };
    },
    async fetchHost(hostId: string) {
      const version = (this.fetchVersions[hostId] = (this.fetchVersions[hostId] ?? 0) + 1);
      try {
        const target = this.targetFor(hostId);
        const listing = await apiJsonTo<ChainJobListing>(target, "/api/chain-jobs");
        if (version !== this.fetchVersions[hostId]) return;
        this.byHost[hostId] = {
          jobs: listing.jobs.sort((a, b) => b.created_at_unix_ms - a.created_at_unix_ms),
          error: null,
        };
      } catch (err) {
        if (version !== this.fetchVersions[hostId]) return;
        // Keep the last listing visible; the poll self-heals transient drops.
        this.byHost[hostId] = { jobs: this.byHost[hostId]?.jobs ?? [], error: String(err) };
      }
      this.syncPolling();
    },
    /** Refresh every connected host and drop listings for gone hosts. */
    async fetchAll() {
      const hosts = useHostsStore();
      const ready = hosts.all.filter((h) => h.status === "ready" && h.baseUrl);
      await Promise.all(ready.map((host) => this.fetchHost(host.id)));
      const live = new Set(hosts.all.map((h) => h.id));
      for (const id of Object.keys(this.byHost)) {
        if (!live.has(id)) delete this.byHost[id];
      }
      this.syncPolling();
    },
    /** Chains land from any surface (CLI, web, another machine) — poll while
     *  anything is queued or running so the strip stays honest. */
    syncPolling() {
      if (this.anyActive && !this.pollTimer) {
        this.pollTimer = setInterval(() => void this.fetchAll(), POLL_INTERVAL_MS);
      } else if (!this.anyActive && this.pollTimer) {
        clearInterval(this.pollTimer);
        this.pollTimer = null;
      }
    },
    async create(hostId: string, req: ChainRequestWire): Promise<string> {
      const target = this.targetFor(hostId);
      const res = await apiJsonTo<CreateChainJobResponse>(target, "/api/chain-jobs", jsonInit(req));
      await this.fetchHost(hostId);
      this.watch(hostId, res.job_id);
      return res.job_id;
    },
    async fetchDetail(hostId: string, jobId: string): Promise<ChainJobDetail> {
      const target = this.targetFor(hostId);
      return apiJsonTo<ChainJobDetail>(target, `/api/chain-jobs/${encodeURIComponent(jobId)}`);
    },
    /** Subscribe to one job's event stream, replacing any prior watch. */
    watch(hostId: string, jobId: string) {
      const target = this.targetFor(hostId);
      this.unwatch();
      this.watching = { hostId, jobId };
      this.live = emptyChainJobLive();
      const abort = new AbortController();
      this.abort = abort;
      void sseStream(`/api/chain-jobs/${encodeURIComponent(jobId)}/events`, {
        signal: abort.signal,
        retry: true,
        target,
        onEvent: (_event, data) => {
          // A superseded stream (new watch, abort still in flight) must not
          // clobber the live view; the abort flag is the ground truth.
          if (abort.signal.aborted) return;
          try {
            const ev = JSON.parse(data) as ChainJobEvent;
            this.live = applyChainJobEvent(this.live, ev);
            if (ev.type === "finalized") this.finalizedTick += 1;
            if (ev.type === "state_changed") this.onStateChanged(hostId, ev.state);
          } catch {
            /* skip malformed frame */
          }
        },
      });
    },
    unwatch() {
      this.abort?.abort();
      this.abort = null;
      this.watching = null;
    },
    onStateChanged(hostId: string, state: ChainJobSummary["state"]) {
      void this.fetchHost(hostId);
      if (state === "completed") {
        const frames =
          this.live.detail?.stages.reduce((n, s) => n + (s.frames_emitted ?? 0), 0) ?? 0;
        // Verb→noun (§11): the sequence is a thing now, and it has a home.
        useToastStore().push("Sequence ready — saved to Library");
        notifyChainFinished(frames);
      } else if (state === "failed") {
        useToastStore().push("Sequence failed", "error");
      }
    },
    async resume(hostId: string, jobId: string) {
      const target = this.targetFor(hostId);
      await apiFetchTo(target, `/api/chain-jobs/${encodeURIComponent(jobId)}/resume`, {
        method: "POST",
      });
      await this.fetchHost(hostId);
      this.watch(hostId, jobId);
    },
    async cancel(hostId: string, jobId: string) {
      const target = this.targetFor(hostId);
      await apiFetchTo(target, `/api/chain-jobs/${encodeURIComponent(jobId)}/cancel`, {
        method: "POST",
      });
      await this.fetchHost(hostId);
    },
    async remove(hostId: string, jobId: string) {
      const target = this.targetFor(hostId);
      await apiFetchTo(target, `/api/chain-jobs/${encodeURIComponent(jobId)}`, {
        method: "DELETE",
      });
      if (this.watching?.hostId === hostId && this.watching.jobId === jobId) this.unwatch();
      const entry = this.byHost[hostId];
      if (entry) entry.jobs = entry.jobs.filter((j) => j.id !== jobId);
    },
    async retake(hostId: string, jobId: string, req: RetakeRequest) {
      const target = this.targetFor(hostId);
      await apiFetchTo(
        target,
        `/api/chain-jobs/${encodeURIComponent(jobId)}/retake`,
        jsonInit(req),
      );
      await this.fetchHost(hostId);
      this.watch(hostId, jobId);
    },
    /** Edit a durable job in place: the server re-renders only dirty stages. */
    async amend(hostId: string, jobId: string, req: AmendRequest): Promise<AmendResponse> {
      const target = this.targetFor(hostId);
      const outcome = await apiJsonTo<AmendResponse>(
        target,
        `/api/chain-jobs/${encodeURIComponent(jobId)}/amend`,
        jsonInit(req),
      );
      await this.fetchHost(hostId);
      this.watch(hostId, jobId);
      return outcome;
    },
    /** Delete every inactive job (anything not running or queued) — on one
     *  host, or every host when none is given. Partial failures resync the
     *  affected host rather than guessing. */
    async clearInactive(hostId?: string): Promise<{ cleared: number; failed: number }> {
      const hostIds = hostId ? [hostId] : Object.keys(this.byHost);
      const doomed = hostIds.flatMap((id) =>
        (this.byHost[id]?.jobs ?? [])
          .filter((job) => !isActive(job))
          .map((job) => ({ hostId: id, jobId: job.id })),
      );
      const results = await Promise.allSettled(
        doomed.map(({ hostId: id, jobId }) => this.remove(id, jobId)),
      );
      const cleared = results.filter((r) => r.status === "fulfilled").length;
      const failed = results.length - cleared;
      if (failed > 0) {
        const failedHosts = new Set(
          doomed.filter((_, i) => results[i]?.status === "rejected").map((d) => d.hostId),
        );
        await Promise.all([...failedHosts].map((id) => this.fetchHost(id)));
      }
      return { cleared, failed };
    },
    async gc(hostId: string): Promise<GcOutcome> {
      const target = this.targetFor(hostId);
      const outcome = await apiJsonTo<GcOutcome>(target, "/api/chain-jobs/gc", {
        method: "POST",
      });
      await this.fetchHost(hostId);
      return outcome;
    },
  },
});
