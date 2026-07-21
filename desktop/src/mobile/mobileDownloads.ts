import { defineStore } from "pinia";
import { reactive, shallowReactive } from "vue";
import { apiFetchTo, ApiError } from "../lib/api/client";
import { startCatalogDownload } from "../lib/api/catalog";
import { sseStream } from "../lib/api/sse";
import type { CatalogEntry, DownloadEvent, DownloadJob } from "../lib/api/types";
import { applyDownloadEvent, emptyDownloadsState, type DownloadsState } from "../lib/downloads";
import { expansionPullJobMatchesModel } from "../lib/expansionPull";
import { percent } from "../lib/format";
import { mobileHostTarget, type MobileHost } from "./hosts";

export type MobileDownloadEntry = Pick<CatalogEntry, "id" | "name">;
export type MobilePendingPullPhase = "connecting" | "starting";

export interface MobilePendingPull {
  entryId: string;
  entryName: string;
  hostId: string;
  jobId: string | null;
  phase: MobilePendingPullPhase;
  /** Exact server/credential identity that owns an in-flight POST. */
  targetSignature?: string;
  /** Present only for an exact frozen Generate recovery lease. */
  ownerId?: string;
}

export interface MobilePullStatus {
  label: string;
  phase: MobilePendingPullPhase | "queued" | "active";
}

export interface MobileDownloadEventContext {
  host: MobileHost;
  event: DownloadEvent;
  hadSnapshot: boolean;
  newlySettled: DownloadJob[];
}

export interface MobileDownloadConsumer {
  onEvent?: (context: MobileDownloadEventContext) => void;
  onStreamError?: (host: MobileHost, cause: Error, phase: "connect" | "closed") => void;
}

export type MobilePullResult =
  | { kind: "started"; jobId: string }
  | { kind: "conflict"; jobId: string }
  | { kind: "existing" }
  | { kind: "adopted" };

interface ConsumerRegistration extends MobileDownloadConsumer {
  hosts: Map<string, MobileHost>;
}

interface HostIdentitySnapshot {
  hostId: string;
  baseUrl: string;
  instanceId: string | null;
}

interface StreamSubscription {
  identity: HostIdentitySnapshot;
  signature: string;
  controller: AbortController;
  ready: Promise<void>;
  close: (notify?: boolean) => void;
}

interface CancellationRecord {
  identity: HostIdentitySnapshot;
  promise: Promise<void>;
}

interface RateSample {
  at: number;
  bytes: number;
}

interface FrozenPullLease {
  ownerId: string;
  host: MobileHost;
  signature: string;
  jobId: string | null;
}

type TerminalDownloadEvent = Extract<
  DownloadEvent,
  { type: "job_done" | "job_failed" | "job_cancelled" }
>;

const STREAM_OPEN_TIMEOUT_MS = 10_000;

function hostIdentity(host: MobileHost): HostIdentitySnapshot {
  return {
    hostId: host.id,
    baseUrl: host.baseUrl,
    instanceId: host.instanceId ?? null,
  };
}

function hostSignature(host: MobileHost): string {
  const identity = hostIdentity(host);
  return `${identity.baseUrl}|${host.apiKey}|${identity.instanceId ?? ""}`;
}

function identitiesAreCompatible(
  current: HostIdentitySnapshot,
  replacement: HostIdentitySnapshot,
): boolean {
  if (current.hostId !== replacement.hostId || current.baseUrl !== replacement.baseUrl)
    return false;
  // A newly discovered instance id enriches the same endpoint, but two known,
  // different ids mean the URL now reaches a different Mold installation.
  return !(
    current.instanceId &&
    replacement.instanceId &&
    current.instanceId !== replacement.instanceId
  );
}

function pullKey(entry: MobileDownloadEntry, hostId: string): string {
  return `${hostId}:${entry.id}`;
}

function manifestNamesAreCompatible(left: string, right: string): boolean {
  return (
    left === right ||
    (!left.includes(":") && right.startsWith(`${left}:`)) ||
    (!right.includes(":") && left.startsWith(`${right}:`))
  );
}

function pullEntriesAreCompatible(left: MobileDownloadEntry, right: MobileDownloadEntry): boolean {
  return [left.id, left.name].some((candidate) =>
    [right.id, right.name].some((other) => manifestNamesAreCompatible(candidate, other)),
  );
}

function isTerminal(
  event: DownloadEvent,
): event is Extract<DownloadEvent, { type: "job_done" | "job_failed" | "job_cancelled" }> {
  return event.type === "job_done" || event.type === "job_failed" || event.type === "job_cancelled";
}

function conflictJobId(cause: unknown): string | null {
  if (!(cause instanceof ApiError) || cause.status !== 409) return null;
  if (typeof cause.body !== "object" || cause.body === null || !("id" in cause.body)) return null;
  const id = (cause.body as { id?: unknown }).id;
  return typeof id === "string" && id.trim() ? id.trim() : null;
}

export const useMobileDownloadsStore = defineStore("mobile-downloads", () => {
  const downloadsByHost = reactive<Record<string, DownloadsState>>({});
  const rateSamplesByHost = reactive<Record<string, Record<string, RateSample[]>>>({});
  const pendingPulls = reactive(new Map<string, MobilePendingPull>());

  const subscriptions = new Map<string, StreamSubscription>();
  const stateIdentityByHost = new Map<string, HostIdentitySnapshot>();
  const terminalJobsByHost = new Map<string, Set<string>>();
  const terminalEventsByHost = new Map<string, Map<string, TerminalDownloadEvent>>();
  const consumers = new Map<string, ConsumerRegistration>();
  const frozenPullLeases = new Map<string, FrozenPullLease>();
  const pendingPullRequests = new WeakMap<MobilePendingPull, Promise<MobilePullResult>>();
  const cancellationPromises = shallowReactive(new Map<string, CancellationRecord[]>());

  function notifyEvent(context: MobileDownloadEventContext): void {
    for (const consumer of consumers.values()) {
      if (consumer.hosts.has(context.host.id)) consumer.onEvent?.(context);
    }
  }

  function notifyStreamError(host: MobileHost, cause: Error, phase: "connect" | "closed"): void {
    for (const consumer of consumers.values()) {
      if (consumer.hosts.has(host.id)) consumer.onStreamError?.(host, cause, phase);
    }
  }

  function pullJobMatches(
    entry: MobileDownloadEntry,
    job: DownloadJob,
    pending?: MobilePendingPull,
  ): boolean {
    if (pending?.jobId) return pending.jobId === job.id;
    if (pending && pending.phase !== "connecting") return false;
    return (
      job.catalog_id === entry.id ||
      job.model === entry.name ||
      expansionPullJobMatchesModel(job, entry.id)
    );
  }

  function jobsForHost(hostId: string): DownloadJob[] {
    const state = downloadsByHost[hostId];
    return state ? [...state.activeJobs, ...state.queued] : [];
  }

  function pullStatusForHost(entry: MobileDownloadEntry, hostId: string): MobilePullStatus | null {
    const pending = pendingPulls.get(pullKey(entry, hostId));
    const job = jobsForHost(hostId).find((candidate) => pullJobMatches(entry, candidate, pending));
    if (job?.status === "active") {
      const progress =
        job.bytes_total > 0 ? ` ${Math.round(percent(job.bytes_done, job.bytes_total))}%` : "";
      return { label: `Pulling${progress}`, phase: "active" };
    }
    if (job?.status === "queued") return { label: "Queued", phase: "queued" };
    if (pending) {
      return {
        label: pending.phase === "connecting" ? "Connecting…" : "Starting…",
        phase: pending.phase,
      };
    }
    return null;
  }

  function reconcilePendingPulls(hostId: string, event?: DownloadEvent): void {
    const state = downloadsByHost[hostId];
    if (!state) return;
    const inFlight = [...state.activeJobs, ...state.queued];
    const terminalId = event && isTerminal(event) ? event.id : null;
    const terminalIds = terminalJobsByHost.get(hostId);

    for (const [key, pending] of pendingPulls) {
      if (pending.hostId !== hostId) continue;
      const entry = { id: pending.entryId, name: pending.entryName };
      const inFlightMatch = inFlight.some((job) => pullJobMatches(entry, job, pending));
      const canAdoptOpeningSnapshot =
        event?.type === "snapshot" && pending.phase === "connecting" && pending.jobId == null;
      if (
        // An opening snapshot can adopt a pull before this client posts. After
        // POST, retain the job-id association because delta model names may be
        // canonicalized away from both the catalog id and display name.
        (canAdoptOpeningSnapshot && inFlightMatch) ||
        (pending.jobId != null && state.history.some((job) => job.id === pending.jobId)) ||
        (pending.jobId != null && terminalIds?.has(pending.jobId)) ||
        (terminalId != null && pending.jobId === terminalId)
      ) {
        pendingPulls.delete(key);
      }
    }
  }

  function clearPendingPullsForHost(hostId: string): void {
    for (const [key, pending] of pendingPulls) {
      if (pending.hostId === hostId) pendingPulls.delete(key);
    }
  }

  function resetHostLifecycle(hostId: string, removeState: boolean): void {
    if (removeState) delete downloadsByHost[hostId];
    else downloadsByHost[hostId] = emptyDownloadsState();
    stateIdentityByHost.delete(hostId);
    terminalJobsByHost.delete(hostId);
    terminalEventsByHost.delete(hostId);
    delete rateSamplesByHost[hostId];
    clearPendingPullsForHost(hostId);
  }

  function reduce(host: MobileHost, event: DownloadEvent): void {
    downloadsByHost[host.id] = applyDownloadEvent(
      downloadsByHost[host.id] ?? emptyDownloadsState(),
      event,
    );

    const rates = (rateSamplesByHost[host.id] ??= {});
    if (event.type === "progress") {
      const samples = (rates[event.id] ??= []);
      const at = Date.now();
      samples.push({ at, bytes: event.bytes_done });
      while (samples.length > 0 && at - samples[0]!.at > 10_000) samples.shift();
    } else if (isTerminal(event)) {
      delete rates[event.id];
    } else if (event.type === "snapshot") {
      const live = new Set(
        [...downloadsByHost[host.id]!.activeJobs, ...downloadsByHost[host.id]!.queued].map(
          (job) => job.id,
        ),
      );
      for (const id of Object.keys(rates)) if (!live.has(id)) delete rates[id];
    }

    let hadSnapshot = terminalJobsByHost.has(host.id);
    let newlySettled: DownloadJob[] = [];
    if (isTerminal(event)) {
      const known = terminalJobsByHost.get(host.id) ?? new Set<string>();
      known.add(event.id);
      terminalJobsByHost.set(host.id, known);
      const events = terminalEventsByHost.get(host.id) ?? new Map<string, TerminalDownloadEvent>();
      events.set(event.id, event);
      while (events.size > 64) events.delete(events.keys().next().value!);
      terminalEventsByHost.set(host.id, events);
    } else if (event.type === "snapshot") {
      const known = terminalJobsByHost.get(host.id) ?? new Set<string>();
      const settled = event.listing.history.filter(
        (job) => job.status === "completed" || job.status === "failed",
      );
      newlySettled = settled.filter((job) => !known.has(job.id));
      terminalJobsByHost.set(host.id, new Set(settled.map((job) => job.id)));
    }

    reconcilePendingPulls(host.id, event);
    notifyEvent({ host, event, hadSnapshot, newlySettled });
    if (isTerminal(event)) {
      const settledOwners = [...frozenPullLeases.values()]
        .filter((lease) => lease.host.id === host.id && lease.jobId === event.id)
        .map((lease) => lease.ownerId);
      for (const ownerId of settledOwners) releaseFrozenPull(ownerId);
    }
  }

  function ensureSubscription(host: MobileHost): StreamSubscription {
    const identity = hostIdentity(host);
    const signature = hostSignature(host);
    const existing = subscriptions.get(host.id);
    if (existing?.signature === signature && !existing.controller.signal.aborted) {
      return existing;
    }
    const previousIdentity = existing?.identity ?? stateIdentityByHost.get(host.id);
    const preserveLifecycle = previousIdentity
      ? identitiesAreCompatible(previousIdentity, identity)
      : true;
    if (existing) {
      // Supersession is intentional. Reject an unopened stream's waiter so it
      // can follow the replacement, without announcing a connection failure.
      existing.close(false);
      subscriptions.delete(host.id);
    }
    if (!preserveLifecycle) resetHostLifecycle(host.id, false);

    downloadsByHost[host.id] ??= emptyDownloadsState();
    stateIdentityByHost.set(host.id, identity);
    const controller = new AbortController();
    let readySettled = false;
    let openTimer: ReturnType<typeof setTimeout>;
    let resolveReady!: () => void;
    let rejectReady!: (error: Error) => void;
    const ready = new Promise<void>((resolve, reject) => {
      resolveReady = resolve;
      rejectReady = reject;
    });
    void ready.catch(() => {});

    const markReady = () => {
      if (readySettled) return;
      readySettled = true;
      clearTimeout(openTimer);
      resolveReady();
    };
    const failReady = (cause: Error, notify = true) => {
      if (readySettled) return;
      readySettled = true;
      clearTimeout(openTimer);
      controller.abort();
      if (subscriptions.get(host.id)?.controller === controller) subscriptions.delete(host.id);
      rejectReady(cause);
      if (notify) notifyStreamError(host, cause, "connect");
    };
    const close = (notify = true) => {
      if (!readySettled) {
        failReady(new Error(`Download monitoring stopped on ${host.name}.`), notify);
        return;
      }
      controller.abort();
    };

    openTimer = setTimeout(
      () => failReady(new Error(`Timed out connecting to downloads on ${host.name}.`)),
      STREAM_OPEN_TIMEOUT_MS,
    );
    const subscription = { identity, signature, controller, ready, close };
    subscriptions.set(host.id, subscription);
    void sseStream("/api/downloads/stream", {
      target: mobileHostTarget(host),
      signal: controller.signal,
      retry: true,
      onOpenError: failReady,
      onClose: (cause) => {
        if (!readySettled) {
          failReady(cause ?? new Error(`Download stream closed on ${host.name}.`));
          return;
        }
        if (subscriptions.get(host.id)?.controller === controller) subscriptions.delete(host.id);
        if (cause && !controller.signal.aborted) {
          clearPendingPullsForHost(host.id);
          notifyStreamError(host, cause, "closed");
          const owners = [...frozenPullLeases.values()]
            .filter((lease) => lease.host.id === host.id)
            .map((lease) => lease.ownerId);
          for (const ownerId of owners) releaseFrozenPull(ownerId);
        }
      },
      onEvent: (_event, data) => {
        if (controller.signal.aborted) return;
        try {
          const event = JSON.parse(data) as DownloadEvent;
          reduce(host, event);
          // Transport-open is insufficient: readiness means the opening server
          // snapshot has already become authoritative for duplicate detection.
          if (event.type === "snapshot") markReady();
        } catch {
          // A later snapshot or delta repairs malformed-frame gaps.
        }
      },
    });
    return subscription;
  }

  function ensureDownloadStream(host: MobileHost): Promise<void> {
    return ensureSubscription(host).ready;
  }

  function etaFor(hostId: string, jobId: string): number | null {
    const job = downloadsByHost[hostId]?.activeJobs.find((candidate) => candidate.id === jobId);
    const samples = rateSamplesByHost[hostId]?.[jobId] ?? [];
    if (!job || samples.length < 2) return null;
    const first = samples[0]!;
    const last = samples.at(-1)!;
    const elapsed = last.at - first.at;
    const transferred = last.bytes - first.bytes;
    if (elapsed <= 0 || transferred <= 0) return null;
    return Math.round(
      Math.max(0, job.bytes_total - last.bytes) / ((transferred * 1_000) / elapsed),
    );
  }

  function terminalJobFor(
    entry: MobileDownloadEntry,
    hostId: string,
    jobId: string,
  ): DownloadJob | null {
    const recorded = downloadsByHost[hostId]?.history.find((job) => job.id === jobId);
    if (recorded) return { ...recorded };
    const event = terminalEventsByHost.get(hostId)?.get(jobId);
    if (!event) return null;
    return {
      id: jobId,
      model: event.type === "job_done" ? event.model : entry.name,
      status:
        event.type === "job_done"
          ? "completed"
          : event.type === "job_failed"
            ? "failed"
            : "cancelled",
      files_done: 0,
      files_total: 0,
      bytes_done: 0,
      bytes_total: 0,
      error: event.type === "job_failed" ? event.error : null,
    };
  }

  function desiredHosts(): Map<string, MobileHost> {
    const desired = new Map<string, MobileHost>();
    for (const consumer of consumers.values()) {
      for (const host of consumer.hosts.values()) desired.set(host.id, host);
    }
    // An explicit Generate recovery is stronger than freshness ordering among
    // ordinary Catalog/host-detail consumers. It owns one exact credential
    // snapshot until the attempt is released.
    for (const lease of frozenPullLeases.values()) desired.set(lease.host.id, lease.host);
    return desired;
  }

  function syncSubscriptions(): void {
    const desired = desiredHosts();
    const knownHostIds = new Set([
      ...subscriptions.keys(),
      ...stateIdentityByHost.keys(),
      ...Object.keys(downloadsByHost),
      ...terminalJobsByHost.keys(),
      ...[...pendingPulls.values()].map((pending) => pending.hostId),
    ]);
    for (const hostId of knownHostIds) {
      if (desired.has(hostId)) continue;
      subscriptions.get(hostId)?.close(false);
      subscriptions.delete(hostId);
      resetHostLifecycle(hostId, true);
    }
    for (const host of desired.values()) void ensureDownloadStream(host).catch(() => {});
  }

  function registerConsumer(
    id: string,
    hosts: MobileHost[],
    listener: MobileDownloadConsumer = {},
  ): void {
    // Map insertion order is the freshness policy for conflicting snapshots.
    // Reinsert existing consumers so the latest lease deterministically wins.
    consumers.delete(id);
    consumers.set(id, {
      ...listener,
      hosts: new Map(hosts.map((host) => [host.id, host])),
    });
    syncSubscriptions();
  }

  function unregisterConsumer(id: string): void {
    consumers.delete(id);
    syncSubscriptions();
  }

  async function awaitWinningStream(
    host: MobileHost,
    key: string,
    pending: MobilePendingPull,
  ): Promise<MobileHost> {
    let candidate = desiredHosts().get(host.id) ?? host;
    while (true) {
      const attempted = ensureSubscription(candidate);
      try {
        await attempted.ready;
      } catch (cause) {
        if (pendingPulls.get(key) !== pending) throw cause;
        const replacement = subscriptions.get(host.id);
        const latest = desiredHosts().get(host.id);
        if (!replacement || !latest || replacement === attempted) throw cause;
        candidate = latest;
        continue;
      }

      if (pendingPulls.get(key) !== pending) return candidate;
      const current = subscriptions.get(host.id);
      const latest = desiredHosts().get(host.id);
      if (current !== attempted) {
        if (!latest) throw new Error(`Download monitoring stopped on ${host.name}.`);
        candidate = latest;
        continue;
      }
      return latest ?? candidate;
    }
  }

  async function startPullInternal(
    entry: MobileDownloadEntry,
    host: MobileHost,
    ownerId?: string,
  ): Promise<MobilePullResult> {
    // Consumers can hold different snapshots of one host while credentials
    // rotate. The authority's latest lease owns both monitoring and the POST;
    // only lease-free direct callers may supply their target verbatim.
    if (!ownerId && pullStatusForHost(entry, host.id)) return { kind: "existing" };
    if (ownerId) {
      const signature = hostSignature(host);
      const shared = [...pendingPulls.values()].find(
        (pending) =>
          pending.hostId === host.id &&
          pending.phase === "starting" &&
          pending.targetSignature === signature &&
          pullEntriesAreCompatible(entry, {
            id: pending.entryId,
            name: pending.entryName,
          }),
      );
      const request = shared ? pendingPullRequests.get(shared) : undefined;
      if (request) return request;
    }
    const key = pullKey(entry, host.id);
    const pending = reactive<MobilePendingPull>({
      entryId: entry.id,
      entryName: entry.name,
      hostId: host.id,
      jobId: null,
      phase: "connecting",
      ...(ownerId ? { ownerId } : {}),
    });
    pendingPulls.set(key, pending);

    const request = (async (): Promise<MobilePullResult> => {
      try {
        const pullHost = await awaitWinningStream(host, key, pending);
        if (pendingPulls.get(key) !== pending) return { kind: "adopted" };
        if (ownerId) {
          const lease = frozenPullLeases.get(host.id);
          if (
            lease?.ownerId !== ownerId ||
            lease.signature !== hostSignature(host) ||
            hostSignature(pullHost) !== lease.signature
          ) {
            throw new Error(`The frozen download route for ${host.name} changed.`);
          }
        }
        pending.targetSignature = hostSignature(pullHost);
        pending.phase = "starting";
        let jobId: string | null;
        let kind: "started" | "conflict" = "started";
        try {
          jobId = await startCatalogDownload(entry.id, mobileHostTarget(pullHost), false);
        } catch (cause) {
          const existingJobId = conflictJobId(cause);
          if (!existingJobId || pendingPulls.get(key) !== pending) throw cause;
          jobId = existingJobId;
          kind = "conflict";
        }
        if (pendingPulls.get(key) === pending) {
          if (jobId === null) throw new Error(`${pullHost.name} did not return a download job.`);
          pending.jobId = jobId;
          reconcilePendingPulls(host.id);
        }
        return jobId === null ? { kind: "adopted" } : { kind, jobId };
      } catch (cause) {
        if (pendingPulls.get(key) === pending) pendingPulls.delete(key);
        throw cause;
      }
    })();
    pendingPullRequests.set(pending, request);
    return request;
  }

  function startPull(entry: MobileDownloadEntry, host: MobileHost): Promise<MobilePullResult> {
    return startPullInternal(entry, host);
  }

  function startPullFrozen(
    entry: MobileDownloadEntry,
    host: MobileHost,
    ownerId: string,
  ): Promise<MobilePullResult> {
    const existing = frozenPullLeases.get(host.id);
    const signature = hostSignature(host);
    if (existing && (existing.ownerId !== ownerId || existing.signature !== signature)) {
      return Promise.reject(
        new Error(`Another frozen download attempt already owns ${host.name}.`),
      );
    }
    frozenPullLeases.set(host.id, {
      ownerId,
      host: { ...host },
      signature,
      jobId: null,
    });
    syncSubscriptions();
    return startPullInternal(entry, host, ownerId)
      .then((result) => {
        const lease = frozenPullLeases.get(host.id);
        if (lease?.ownerId === ownerId) {
          lease.jobId =
            result.kind === "started" || result.kind === "conflict"
              ? result.jobId
              : (jobsForHost(host.id).find((job) => pullJobMatches(entry, job))?.id ?? null);
          if (
            lease.jobId &&
            (terminalJobsByHost.get(host.id)?.has(lease.jobId) ||
              terminalJobFor(entry, host.id, lease.jobId))
          ) {
            releaseFrozenPull(ownerId);
          }
        }
        return result;
      })
      .catch((cause) => {
        releaseFrozenPull(ownerId);
        throw cause;
      });
  }

  function releaseFrozenPull(ownerId: string): void {
    for (const [hostId, lease] of frozenPullLeases) {
      if (lease.ownerId === ownerId) frozenPullLeases.delete(hostId);
    }
    for (const [key, pending] of pendingPulls) {
      if (pending.ownerId === ownerId) pendingPulls.delete(key);
    }
    syncSubscriptions();
  }

  function isCancelling(host: MobileHost, jobId: string): boolean {
    const currentHost = desiredHosts().get(host.id) ?? host;
    const identity = hostIdentity(currentHost);
    const key = `${host.id}:${jobId}:${identity.baseUrl}`;
    return (cancellationPromises.get(key) ?? []).some((record) =>
      identitiesAreCompatible(record.identity, identity),
    );
  }

  function cancel(host: MobileHost, job: DownloadJob): Promise<void> {
    const cancelHost = desiredHosts().get(host.id) ?? host;
    const identity = hostIdentity(cancelHost);
    const key = `${host.id}:${job.id}:${identity.baseUrl}`;
    const records = cancellationPromises.get(key) ?? [];
    const existing = records.find((record) => identitiesAreCompatible(record.identity, identity));
    if (existing) return existing.promise;

    let record!: CancellationRecord;
    const request = apiFetchTo(
      mobileHostTarget(cancelHost),
      `/api/downloads/${encodeURIComponent(job.id)}`,
      { method: "DELETE" },
    )
      .then(() => undefined)
      .finally(() => {
        const remaining = (cancellationPromises.get(key) ?? []).filter(
          (candidate) => candidate !== record,
        );
        if (remaining.length) cancellationPromises.set(key, remaining);
        else cancellationPromises.delete(key);
      });
    record = { identity, promise: request };
    cancellationPromises.set(key, [...records, record]);
    return request;
  }

  return {
    downloadsByHost,
    pendingPulls,
    registerConsumer,
    unregisterConsumer,
    ensureDownloadStream,
    etaFor,
    terminalJobFor,
    pullStatusForHost,
    startPull,
    startPullFrozen,
    releaseFrozenPull,
    cancel,
    isCancelling,
  };
});
