import { defineStore } from "pinia";
import { apiFetch, apiFetchTo, apiJson, apiJsonTo, type ApiTarget } from "../lib/api/client";
import { sseStream } from "../lib/api/sse";
import { notifyChainFinished } from "../lib/notify";
import { useToastStore } from "./toasts";
import type {
  ChainJobDetail,
  ChainJobEvent,
  ChainJobListing,
  ChainJobStageDetail,
  ChainJobSummary,
  ChainRequest,
  CreateChainJobResponse,
  GcOutcome,
  RetakeRequest,
  StageState,
} from "../lib/api/types";

/** Live view of the job currently being watched: its detail plus per-stage
 * denoise progress and which stage is active. */
export interface ChainJobLive {
  detail: ChainJobDetail | null;
  progress: Record<number, { step: number; total: number }>;
  activeStage: number | null;
}

export function emptyChainJobLive(): ChainJobLive {
  return { detail: null, progress: {}, activeStage: null };
}

const sameTarget = (a: ApiTarget | null, b: ApiTarget | null) =>
  a?.baseUrl === b?.baseUrl && a?.apiKey === b?.apiKey;

function firstRunningStage(job: ChainJobDetail): number | null {
  const running = job.stages.find((s) => s.state === "running");
  return running ? running.idx : null;
}

function withStageState(
  detail: ChainJobDetail | null,
  idx: number,
  state: StageState,
  hasPreview?: boolean,
): ChainJobDetail | null {
  if (!detail) return detail;
  const stages = detail.stages.map((s): ChainJobStageDetail =>
    s.idx === idx ? { ...s, state, has_preview: hasPreview ?? s.has_preview } : s,
  );
  return { ...detail, stages };
}

/**
 * Pure chain-job event reducer. First frame is a `snapshot` (full detail);
 * deltas advance the active stage's state and denoise progress. Exported for
 * tests.
 */
export function applyChainJobEvent(state: ChainJobLive, ev: ChainJobEvent): ChainJobLive {
  switch (ev.type) {
    case "snapshot":
      return { detail: ev.job, progress: {}, activeStage: firstRunningStage(ev.job) };
    case "stage_start":
      return {
        ...state,
        activeStage: ev.stage_idx,
        detail: withStageState(state.detail, ev.stage_idx, "running"),
      };
    case "denoise_step":
      return {
        ...state,
        progress: { ...state.progress, [ev.stage_idx]: { step: ev.step, total: ev.total } },
      };
    case "stage_done":
      return {
        ...state,
        activeStage: state.activeStage === ev.stage_idx ? null : state.activeStage,
        detail: withStageState(state.detail, ev.stage_idx, "completed", ev.has_preview),
      };
    case "state_changed":
      return {
        ...state,
        detail: state.detail ? { ...state.detail, state: ev.state, error: ev.error ?? null } : null,
      };
    default:
      return state;
  }
}

export const useChainJobsStore = defineStore("chainJobs", {
  state: () => ({
    jobs: [] as ChainJobSummary[],
    live: emptyChainJobLive() as ChainJobLive,
    watchingId: null as string | null,
    abort: null as AbortController | null,
    pollTimer: null as ReturnType<typeof setInterval> | null,
    /** Host owning the currently displayed chain-job list and live stream. */
    target: null as ApiTarget | null,
    fetchVersion: 0,
  }),
  actions: {
    async fetchJobs(target?: ApiTarget | null) {
      const resolvedTarget = target === undefined ? this.target : target;
      if (!sameTarget(this.target, resolvedTarget)) {
        this.unwatch();
        this.live = emptyChainJobLive();
      }
      this.target = resolvedTarget;
      const fetchVersion = ++this.fetchVersion;
      const listing = resolvedTarget
        ? await apiJsonTo<ChainJobListing>(resolvedTarget, "/api/chain-jobs")
        : await apiJson<ChainJobListing>("/api/chain-jobs");
      if (fetchVersion !== this.fetchVersion) return;
      // Newest first.
      this.jobs = listing.jobs.sort((a, b) => b.created_at_unix_ms - a.created_at_unix_ms);
      // Chains land from any surface (CLI, web, another machine sharing the
      // DB) — keep the list honest while anything is in flight.
      const active = this.jobs.some((j) => j.state === "running" || j.state === "queued");
      if (active && !this.pollTimer) {
        this.pollTimer = setInterval(() => void this.fetchJobs(), 10_000);
      } else if (!active && this.pollTimer) {
        clearInterval(this.pollTimer);
        this.pollTimer = null;
      }
    },
    async create(req: ChainRequest, target: ApiTarget | null = null): Promise<string> {
      this.target = target;
      const init = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
      } satisfies RequestInit;
      const res = target
        ? await apiJsonTo<CreateChainJobResponse>(target, "/api/chain-jobs", init)
        : await apiJson<CreateChainJobResponse>("/api/chain-jobs", init);
      await this.fetchJobs(target);
      this.watch(res.job_id, target);
      return res.job_id;
    },
    /** Subscribe to one job's event stream, replacing any prior watch. */
    watch(id: string, target?: ApiTarget | null) {
      const resolvedTarget = target === undefined ? this.target : target;
      this.unwatch();
      this.target = resolvedTarget;
      this.watchingId = id;
      this.live = emptyChainJobLive();
      const abort = new AbortController();
      this.abort = abort;
      void sseStream(`/api/chain-jobs/${encodeURIComponent(id)}/events`, {
        signal: abort.signal,
        retry: true,
        ...(resolvedTarget ? { target: resolvedTarget } : {}),
        onEvent: (_event, data) => {
          try {
            const ev = JSON.parse(data) as ChainJobEvent;
            this.live = applyChainJobEvent(this.live, ev);
            if (ev.type === "state_changed") this.onStateChanged(ev.state, ev.error ?? null);
          } catch {
            /* skip malformed frame */
          }
        },
      });
    },
    unwatch() {
      this.abort?.abort();
      this.abort = null;
      this.watchingId = null;
    },
    onStateChanged(state: ChainJobSummary["state"], _error: string | null) {
      void this.fetchJobs();
      if (state === "completed") {
        const frames =
          this.live.detail?.stages.reduce((n, s) => n + (s.frames_emitted ?? 0), 0) ?? 0;
        useToastStore().push(`Chain finished · ${frames} frames`);
        notifyChainFinished(frames);
      } else if (state === "failed") {
        useToastStore().push("Chain failed", "error");
      }
    },
    async resume(id: string) {
      const path = `/api/chain-jobs/${encodeURIComponent(id)}/resume`;
      const init = { method: "POST" } satisfies RequestInit;
      if (this.target) await apiFetchTo(this.target, path, init);
      else await apiFetch(path, init);
      await this.fetchJobs();
      this.watch(id);
    },
    async cancel(id: string) {
      const path = `/api/chain-jobs/${encodeURIComponent(id)}/cancel`;
      const init = { method: "POST" } satisfies RequestInit;
      if (this.target) await apiFetchTo(this.target, path, init);
      else await apiFetch(path, init);
      await this.fetchJobs();
    },
    async remove(id: string) {
      const path = `/api/chain-jobs/${encodeURIComponent(id)}`;
      const init = { method: "DELETE" } satisfies RequestInit;
      if (this.target) await apiFetchTo(this.target, path, init);
      else await apiFetch(path, init);
      if (this.watchingId === id) this.unwatch();
      this.jobs = this.jobs.filter((j) => j.id !== id);
    },
    async retake(id: string, req: RetakeRequest) {
      const path = `/api/chain-jobs/${encodeURIComponent(id)}/retake`;
      const init = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
      } satisfies RequestInit;
      if (this.target) await apiFetchTo(this.target, path, init);
      else await apiFetch(path, init);
      await this.fetchJobs();
      this.watch(id);
    },
    async gc(): Promise<GcOutcome> {
      const init = { method: "POST" } satisfies RequestInit;
      const outcome = this.target
        ? await apiJsonTo<GcOutcome>(this.target, "/api/chain-jobs/gc", init)
        : await apiJson<GcOutcome>("/api/chain-jobs/gc", init);
      await this.fetchJobs();
      return outcome;
    },
  },
});
