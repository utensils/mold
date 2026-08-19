/*
 * Generation host routing for the web Create surface (spec §08 multi-host).
 *
 * TS twin of the desktop app's `desktop/src/lib/hosts.ts` router: the browser
 * has no Pinia store and no Tauri IPC, so the *logic* is ported rather than the
 * code. Every rule here must keep agreeing with desktop — Auto is model-aware
 * least-busy, "Most capable" walks the CUDA > Metal > unknown ladder, and a
 * sticky explicit pick is honoured until the host disappears from the registry.
 *
 * The origin host plays desktop's "local" role in tie-breaks: same-origin
 * dispatch costs no network hop, so it wins a dead heat.
 */
import {
  AUTO_TARGET_ID,
  CAPABLE_TARGET_ID,
  backendRank as sharedBackendRank,
  hostIdsForModel as sharedHostIdsForModel,
  normalizeTargetId as sharedNormalizeTargetId,
  pickAutoHost as sharedPickAutoHost,
  pickMostCapableHost as sharedPickMostCapableHost,
  unionModelsByName,
  type HostRoutingStatus,
} from "@studio/lib/hostRouting";
import { ORIGIN_HOST_ID } from "./hostRegistry";
import type { ModelInfoExtended } from "../types";
import type { ReferenceUploadCapabilities } from "@studio/api/referenceUploads";

// The routing policy itself lives in `@studio/lib/hostRouting`, shared with
// desktop and the iPhone app; this module binds it to the browser registry,
// whose home host is the serving origin.
export { AUTO_TARGET_ID, CAPABLE_TARGET_ID, type HostRoutingStatus };

/** GPU summary as the current `/api/status` contract reports it. */
export interface RoutableGpu {
  backend: string | null;
  name?: string | null;
  vramTotalMb?: number | null;
}

/** The slice of a registry host the routers reason over. */
export interface RoutableHost {
  id: string;
  label: string;
  /** Origin URL used as the request base; never carries a key. */
  url: string;
  /** Per-host key, sent as `x-api-key`. Never placed in a URL. */
  apiKey?: string;
  instanceId?: string;
  status: HostRoutingStatus;
  /** Live queue depth; null while unknown (counts as busiest). */
  queueDepth: number | null;
  /** Predicted end of this host's current plan. Null on legacy hosts. */
  predictedCompletionMs?: number | null;
  gpu: RoutableGpu | null;
}

/** Where a submission is dispatched: the host's base URL plus its key. */
export interface HostTarget {
  baseUrl: string;
  apiKey?: string;
}

export interface HostRoute {
  hostId: string;
  label: string;
  target: HostTarget;
  /** Frozen server-installation identity for same-endpoint replacement fences. */
  instanceId?: string | null;
  /** Frozen authenticated reference-ingress contract for this exact host. */
  referenceUploads?: ReferenceUploadCapabilities | null;
}

export function sameHostRoute(
  frozen: HostRoute | null,
  current: HostRoute | null,
): boolean {
  if (!frozen) return !current || current.hostId === ORIGIN_HOST_ID;
  if (!current) return false;
  return (
    frozen.hostId === current.hostId &&
    frozen.target.baseUrl === current.target.baseUrl &&
    frozen.target.apiKey === current.target.apiKey &&
    (frozen.instanceId ?? null) === (current.instanceId ?? null)
  );
}

function isOrigin(host: { id: string }): boolean {
  return host.id === ORIGIN_HOST_ID;
}

/**
 * Auto routing: when both hosts expose an authoritative plan, predicted
 * completion wins before raw queue depth. If either host is planless, queue
 * depth is the deterministic backward-compatible fallback.
 */
export function pickAutoHost<T extends RoutableHost>(hosts: readonly T[]): T | null {
  return sharedPickAutoHost(hosts, { isHome: isOrigin, lowestIdWins: true });
}

/** Capability ladder: CUDA (2) > Metal (1) > CPU/unknown (0). */
export function backendRank(backend: string | null): number {
  return sharedBackendRank(backend);
}

/**
 * "Most capable" routing: among ready hosts — restricted to the ones that
 * already hold the model when `modelHostIds` says at least one ready host does
 * — rank by backend, then total VRAM descending, then queue depth ascending. A
 * fully tied origin wins. Null when nothing is ready.
 */
export function pickMostCapableHost<T extends RoutableHost>(
  hosts: readonly T[],
  modelHostIds: readonly string[] | null,
): T | null {
  return sharedPickMostCapableHost(hosts, modelHostIds, { isHome: isOrigin, lowestIdWins: true });
}

/**
 * Normalize the persisted target the way the picker displays it: the sentinels
 * and a currently-listed host id pass through; a ghost id (the host was
 * forgotten) reads as Auto. Every consumer must read the pref through this so
 * the picker's selection and the dispatch decision can never disagree.
 */
export function normalizeTargetId(
  selection: string | null | undefined,
  hosts: ReadonlyArray<{ id: string }>,
): string {
  return sharedNormalizeTargetId(selection, hosts);
}

/**
 * Resolve the concrete host a submission dispatches to.
 *
 * A sticky pick that is listed but not ready resolves to null — the caller must
 * surface that as an error rather than silently rerouting the user's print to a
 * different machine (the desktop rule). A pick that is *gone* from the registry
 * has already degraded to Auto via `normalizeTargetId`.
 */
export function resolveRoute(
  hosts: readonly RoutableHost[],
  selection: string | null | undefined,
  modelHostIds: readonly string[] = [],
): HostRoute | null {
  const sel = normalizeTargetId(selection, hosts);
  let chosen: RoutableHost | null;
  if (sel === CAPABLE_TARGET_ID) {
    chosen = pickMostCapableHost(
      hosts,
      modelHostIds.length > 0 ? modelHostIds : null,
    );
  } else if (sel !== AUTO_TARGET_ID) {
    chosen = hosts.find((h) => h.id === sel && h.status === "ready") ?? null;
  } else {
    const withModel = hosts.filter(
      (h) => h.status === "ready" && modelHostIds.includes(h.id),
    );
    chosen = pickAutoHost(withModel.length > 0 ? withModel : hosts);
  }
  if (!chosen) return null;
  const target: HostTarget = { baseUrl: chosen.url };
  if (chosen.apiKey) target.apiKey = chosen.apiKey;
  return {
    hostId: chosen.id,
    label: chosen.label,
    target,
    instanceId: chosen.instanceId ?? null,
  };
}

/** Per-host `/api/models` snapshots, keyed by registry host id. */
export type ModelsByHost = Record<string, ModelInfoExtended[]>;

/**
 * The model list for a set of hosts, deduped by name. A downloaded copy always
 * beats an undownloaded one — a model installed on any host in the set is
 * installed as far as routing is concerned.
 */
export function unionModels(
  modelsByHost: ModelsByHost,
  hostIds: readonly string[],
): ModelInfoExtended[] {
  return unionModelsByName(modelsByHost, hostIds);
}

/** Ids of the hosts that have `name` downloaded — the model-aware routing input. */
export function hostIdsForModel(modelsByHost: ModelsByHost, name: string): string[] {
  return sharedHostIdsForModel(modelsByHost, name);
}
