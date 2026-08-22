/*
 * Durable sequence (chain) jobs across every known host — the web answer to
 * desktop's chainJobs store, following useGenerateStream's module-singleton
 * pattern so the state survives route changes and every consumer shares it.
 *
 * Targets are resolved AT CALL TIME through the host registry: a renamed key
 * or re-added host is picked up on the next action rather than frozen into
 * the composable. The watch stream reuses useChainJobStream's SSE wiring
 * (@microsoft/fetch-event-source with a 250 ms poll fallback) but reduces
 * frames with the shared @studio applyChainJobEvent reducer.
 */

import { fetchEventSource } from "@microsoft/fetch-event-source";
import { computed, reactive, ref, type ComputedRef, type Ref } from "vue";
import {
  amendChainJob,
  cancelChainJob,
  cancelChainJobMutation,
  chainJobEventsUrl,
  createChainJob,
  deleteChainJob,
  gcChainJobs,
  getChainJob,
  listChainJobs,
  resumeChainJob,
  retakeChainJob,
  type StreamTarget,
} from "../api";
import { listHosts, ORIGIN_HOST_ID } from "../lib/hostRegistry";
import {
  applyChainJobEvent,
  emptyChainJobLive,
  type ChainJobLive,
} from "@studio/lib/chainJobEvents";
import { isAuthoredSequenceJob } from "@studio/lib/api/chainTypes";
import type {
  AmendRequest,
  AmendResponse,
  ChainJobDetail,
  ChainJobEvent,
  ChainJobSummary,
  ChainRequestWire,
} from "@studio/lib/api/chainTypes";
import type { RetakeRequest } from "../types";
import { toast } from "../lib/toasts";

export interface ChainJobsHostState {
  jobs: ChainJobSummary[];
  error: string | null;
}

export interface TrackedSequence {
  hostId: string;
  jobId: string;
}

export interface ChainJobsState {
  byHost: Record<string, ChainJobsHostState>;
  watching: TrackedSequence | null;
  live: ChainJobLive;
}

export const TRACKED_SEQUENCES_KEY = "mold.create.tracked-sequences.v1";
const LEGACY_JOB_KEY = "mold.create.chain-job";
const LEGACY_HOST_KEY = "mold.create.chain-job-host";
const POLL_INTERVAL_MS = 10_000;
const WATCH_POLL_FALLBACK_MS = 250;

// ── Module-level singleton state ─────────────────────────────────────────────
// reactive() wrap is load-bearing: SSE and timer closures mutate this state,
// and raw-object mutation would bypass the proxy traps (repo gotcha).
const state = reactive<ChainJobsState>({
  byHost: {},
  watching: null,
  live: emptyChainJobLive(),
});

const tracked = ref<TrackedSequence[]>([]);
let loaded = false;
let watchAbort: AbortController | null = null;
let watchPollTimer: ReturnType<typeof setTimeout> | null = null;
let listPollTimer: ReturnType<typeof setTimeout> | null = null;

function persistTracked() {
  try {
    localStorage.setItem(TRACKED_SEQUENCES_KEY, JSON.stringify(tracked.value));
  } catch {
    // Storage is recovery convenience; the durable server job keeps running.
  }
}

/** Load tracked sequences, migrating the legacy single-job key pair. */
function loadTracked(): TrackedSequence[] {
  let entries: TrackedSequence[] = [];
  try {
    const raw = localStorage.getItem(TRACKED_SEQUENCES_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as unknown;
      if (Array.isArray(parsed)) {
        entries = parsed.filter(
          (e): e is TrackedSequence =>
            typeof (e as TrackedSequence)?.hostId === "string" &&
            typeof (e as TrackedSequence)?.jobId === "string",
        );
      }
    }
    const legacyJob = localStorage.getItem(LEGACY_JOB_KEY);
    if (legacyJob) {
      const legacyHost =
        localStorage.getItem(LEGACY_HOST_KEY) ?? ORIGIN_HOST_ID;
      if (!entries.some((e) => e.jobId === legacyJob)) {
        entries.push({ hostId: legacyHost, jobId: legacyJob });
      }
      localStorage.removeItem(LEGACY_JOB_KEY);
      localStorage.removeItem(LEGACY_HOST_KEY);
      localStorage.setItem(TRACKED_SEQUENCES_KEY, JSON.stringify(entries));
    }
  } catch {
    return entries;
  }
  return entries;
}

function ensureLoaded() {
  if (loaded) return;
  loaded = true;
  tracked.value = loadTracked();
}

function track(hostId: string, jobId: string) {
  if (tracked.value.some((e) => e.hostId === hostId && e.jobId === jobId))
    return;
  tracked.value = [...tracked.value, { hostId, jobId }];
  persistTracked();
}

function untrack(hostId: string, jobId: string) {
  tracked.value = tracked.value.filter(
    (e) => !(e.hostId === hostId && e.jobId === jobId),
  );
  persistTracked();
}

/** Resolve a host id to a dispatch target NOW — never from a snapshot. */
function targetFor(hostId: string): StreamTarget | null {
  const host = listHosts().find((h) => h.id === hostId);
  if (!host) return null;
  return { baseUrl: host.url, ...(host.apiKey ? { apiKey: host.apiKey } : {}) };
}

function anyActive(): boolean {
  return Object.values(state.byHost).some((host) =>
    host.jobs.some((j) => j.state === "queued" || j.state === "running"),
  );
}

function stopListPolling() {
  if (listPollTimer) clearTimeout(listPollTimer);
  listPollTimer = null;
}

/** Chains land from any surface (CLI, another browser, desktop) — keep the
 * lists honest on a 10 s cadence while anything is in flight. */
function ensureListPolling() {
  if (!anyActive()) {
    stopListPolling();
    return;
  }
  if (listPollTimer) return;
  listPollTimer = setTimeout(() => {
    listPollTimer = null;
    void fetchAll();
  }, POLL_INTERVAL_MS);
}

async function fetchHost(hostId: string): Promise<void> {
  ensureLoaded();
  const target = targetFor(hostId);
  if (!target) {
    // A forgotten host has no reachable list — drop its stale bucket.
    delete state.byHost[hostId];
    return;
  }
  try {
    const listing = await listChainJobs(target);
    const jobs = listing.jobs
      .filter(isAuthoredSequenceJob)
      .sort((a, b) => b.created_at_unix_ms - a.created_at_unix_ms);
    state.byHost[hostId] = { jobs, error: null };
    // Server-side deletions release our tracked interest.
    const known = new Set(jobs.map((j) => j.id));
    const pruned = tracked.value.filter(
      (e) => e.hostId !== hostId || known.has(e.jobId),
    );
    if (pruned.length !== tracked.value.length) {
      tracked.value = pruned;
      persistTracked();
    }
  } catch (error) {
    state.byHost[hostId] = {
      jobs: state.byHost[hostId]?.jobs ?? [],
      error: error instanceof Error ? error.message : String(error),
    };
  }
  ensureListPolling();
}

/** Every registry host plus any tracked host that still resolves. */
function hostIdsOfInterest(): string[] {
  const ids = new Set(listHosts().map((h) => h.id));
  for (const entry of tracked.value) ids.add(entry.hostId);
  return [...ids];
}

async function fetchAll(): Promise<void> {
  ensureLoaded();
  await Promise.all(hostIdsOfInterest().map((id) => fetchHost(id)));
}

function stopWatchPoll() {
  if (watchPollTimer) clearTimeout(watchPollTimer);
  watchPollTimer = null;
}

function scheduleWatchPoll(hostId: string, jobId: string) {
  stopWatchPoll();
  watchPollTimer = setTimeout(() => {
    watchPollTimer = null;
    const current = state.watching;
    if (!current || current.hostId !== hostId || current.jobId !== jobId)
      return;
    const target = targetFor(hostId) ?? undefined;
    void getChainJob(jobId, target)
      .then((job) => {
        const again = state.watching;
        if (!again || again.jobId !== jobId) return;
        // The web wire detail carries `script.stage` extras the reducer
        // never reads; structurally it IS the studio snapshot payload.
        state.live = applyChainJobEvent(state.live, {
          type: "snapshot",
          job: job as unknown as ChainJobDetail,
        });
      })
      .catch(() => {
        // The SSE reconnect loop remains authoritative.
      });
  }, WATCH_POLL_FALLBACK_MS);
}

function unwatch() {
  watchAbort?.abort();
  watchAbort = null;
  stopWatchPoll();
  state.watching = null;
}

function watch(hostId: string, jobId: string) {
  ensureLoaded();
  unwatch();
  state.watching = { hostId, jobId };
  state.live = emptyChainJobLive();
  const controller = new AbortController();
  watchAbort = controller;
  const target = targetFor(hostId) ?? undefined;
  void fetchEventSource(chainJobEventsUrl(jobId, target), {
    signal: controller.signal,
    openWhenHidden: true,
    headers: target?.apiKey ? { "x-api-key": target.apiKey } : {},
    onmessage: (message) => {
      if (message.event !== "chain_job") return;
      try {
        const ev = JSON.parse(message.data) as ChainJobEvent;
        state.live = applyChainJobEvent(state.live, ev);
        if (ev.type === "state_changed") void fetchHost(hostId);
      } catch {
        // Bad frames are ignored; the poll fallback re-synchronizes.
      }
    },
    onclose: () => {
      if (!controller.signal.aborted)
        throw new Error("Sequence event stream closed.");
    },
    onerror: () => {
      if (!controller.signal.aborted) {
        scheduleWatchPoll(hostId, jobId);
        return 1_000;
      }
    },
  }).catch(() => {
    // Aborted or terminally failed; polling/refetch cover recovery.
  });
}

async function create(
  hostId: string,
  req: ChainRequestWire,
  frozenTarget?: StreamTarget,
  operationId?: string,
): Promise<string> {
  ensureLoaded();
  const target = frozenTarget ?? targetFor(hostId);
  if (!target) throw new Error("That machine is no longer connected.");
  const { job_id } = await createChainJob(
    req,
    target,
    operationId,
    (warnings) => {
      for (const warning of warnings) toast("warning", warning);
    },
  );
  track(hostId, job_id);
  void fetchHost(hostId);
  watch(hostId, job_id);
  return job_id;
}

async function cancel(
  hostId: string,
  jobId: string,
  frozenTarget?: StreamTarget,
): Promise<void> {
  const target = frozenTarget ?? targetFor(hostId) ?? undefined;
  await cancelChainJob(jobId, target);
  await fetchHost(hostId);
}

async function resume(hostId: string, jobId: string): Promise<void> {
  const target = targetFor(hostId) ?? undefined;
  await resumeChainJob(jobId, target);
  await fetchHost(hostId);
  watch(hostId, jobId);
}

async function remove(hostId: string, jobId: string): Promise<void> {
  const target = targetFor(hostId) ?? undefined;
  await deleteChainJob(jobId, target);
  untrack(hostId, jobId);
  if (state.watching?.jobId === jobId) unwatch();
  const host = state.byHost[hostId];
  if (host) host.jobs = host.jobs.filter((j) => j.id !== jobId);
}

async function retake(
  hostId: string,
  jobId: string,
  req: RetakeRequest,
): Promise<void> {
  const target = targetFor(hostId) ?? undefined;
  await retakeChainJob(jobId, req, target);
  await fetchHost(hostId);
  watch(hostId, jobId);
}

/** POST the full edited stage list. Rethrows so the caller can branch on an
 * `ApiHttpError` 409 (job moved on → create the edit as a new sequence). */
async function amend(
  hostId: string,
  jobId: string,
  req: AmendRequest,
  frozenTarget?: StreamTarget,
  operationId?: string,
): Promise<AmendResponse> {
  const target = frozenTarget ?? targetFor(hostId) ?? undefined;
  const response = await amendChainJob(jobId, req, target, operationId);
  track(hostId, jobId);
  void fetchHost(hostId);
  watch(hostId, jobId);
  return response;
}

async function cancelMutation(
  hostId: string,
  jobId: string,
  operationId: string,
  frozenTarget?: StreamTarget,
): Promise<void> {
  const target = frozenTarget ?? targetFor(hostId) ?? undefined;
  await cancelChainJobMutation(jobId, operationId, target);
}

/** Delete every settled job (anything not running or queued) on every host. */
async function clearInactive(): Promise<{ cleared: number; failed: number }> {
  const victims: TrackedSequence[] = [];
  for (const [hostId, host] of Object.entries(state.byHost)) {
    for (const job of host.jobs) {
      if (job.state !== "running" && job.state !== "queued") {
        victims.push({ hostId, jobId: job.id });
      }
    }
  }
  const results = await Promise.allSettled(
    victims.map((v) => remove(v.hostId, v.jobId)),
  );
  const cleared = results.filter((r) => r.status === "fulfilled").length;
  await fetchAll();
  return { cleared, failed: results.length - cleared };
}

/** POST /api/chain-jobs/gc on every reachable host ("Clean up disk"). */
async function gc(): Promise<{
  swept_ephemeral_jobs: number;
  pruned_artifact_dirs: number;
}> {
  ensureLoaded();
  const outcomes = await Promise.allSettled(
    listHosts().map((host) =>
      gcChainJobs({
        baseUrl: host.url,
        ...(host.apiKey ? { apiKey: host.apiKey } : {}),
      }),
    ),
  );
  await fetchAll();
  return outcomes.reduce(
    (sum, r) =>
      r.status === "fulfilled"
        ? {
            swept_ephemeral_jobs:
              sum.swept_ephemeral_jobs + r.value.swept_ephemeral_jobs,
            pruned_artifact_dirs:
              sum.pruned_artifact_dirs + r.value.pruned_artifact_dirs,
          }
        : sum,
    { swept_ephemeral_jobs: 0, pruned_artifact_dirs: 0 },
  );
}

/** Boot: refresh every host of interest, then reattach the watch to the
 * most recently tracked in-flight sequence so a reload never orphans one. */
async function start(): Promise<void> {
  ensureLoaded();
  await fetchAll();
  if (state.watching) return;
  for (const entry of [...tracked.value].reverse()) {
    const job = state.byHost[entry.hostId]?.jobs.find(
      (j) => j.id === entry.jobId,
    );
    if (job && (job.state === "running" || job.state === "queued")) {
      watch(entry.hostId, entry.jobId);
      return;
    }
  }
}

export interface UseChainJobs {
  state: ChainJobsState;
  tracked: Ref<TrackedSequence[]>;
  /** True while any host reports a queued or running sequence. */
  anyActive: ComputedRef<boolean>;
  start: () => Promise<void>;
  fetchHost: (hostId: string) => Promise<void>;
  fetchAll: () => Promise<void>;
  create: (
    hostId: string,
    req: ChainRequestWire,
    frozenTarget?: StreamTarget,
    operationId?: string,
  ) => Promise<string>;
  watch: (hostId: string, jobId: string) => void;
  unwatch: () => void;
  cancel: (
    hostId: string,
    jobId: string,
    frozenTarget?: StreamTarget,
  ) => Promise<void>;
  cancelMutation: (
    hostId: string,
    jobId: string,
    operationId: string,
    frozenTarget?: StreamTarget,
  ) => Promise<void>;
  resume: (hostId: string, jobId: string) => Promise<void>;
  remove: (hostId: string, jobId: string) => Promise<void>;
  retake: (hostId: string, jobId: string, req: RetakeRequest) => Promise<void>;
  amend: (
    hostId: string,
    jobId: string,
    req: AmendRequest,
    frozenTarget?: StreamTarget,
    operationId?: string,
  ) => Promise<AmendResponse>;
  clearInactive: () => Promise<{ cleared: number; failed: number }>;
  gc: () => Promise<{
    swept_ephemeral_jobs: number;
    pruned_artifact_dirs: number;
  }>;
}

const activeComputed = computed(() => anyActive());

export function useChainJobs(): UseChainJobs {
  ensureLoaded();
  return {
    state,
    tracked,
    anyActive: activeComputed,
    start,
    fetchHost,
    fetchAll,
    create,
    watch,
    unwatch,
    cancel,
    cancelMutation,
    resume,
    remove,
    retake,
    amend,
    clearInactive,
    gc,
  };
}

/** Reset the singleton between tests. */
export const __testing__ = {
  POLL_INTERVAL_MS,
  WATCH_POLL_FALLBACK_MS,
  reset(): void {
    unwatch();
    stopListPolling();
    state.byHost = {};
    state.live = emptyChainJobLive();
    tracked.value = [];
    loaded = false;
  },
};
