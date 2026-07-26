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
import { ORIGIN_HOST_ID } from "./hostRegistry";
import type { ModelInfoExtended } from "../types";

/** Least-busy routing across every ready host. */
export const AUTO_TARGET_ID = "auto";
/** Strongest-GPU routing (desktop's `generateTargetHost = "capable"`). */
export const CAPABLE_TARGET_ID = "capable";

export type HostRoutingStatus = "connecting" | "ready" | "error";

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
}

function isOrigin(host: { id: string }): boolean {
  return host.id === ORIGIN_HOST_ID;
}

function depth(host: RoutableHost): number {
  return host.queueDepth ?? Number.MAX_SAFE_INTEGER;
}

/**
 * Auto routing: the ready host with the shallowest queue wins. Predictions
 * refine equal-depth hosts only when both are known, so a planless current,
 * legacy, or observe-mode host is never starved by rollout asymmetry.
 */
export function pickAutoHost<T extends RoutableHost>(
  hosts: readonly T[],
): T | null {
  const ready = hosts.filter((h) => h.status === "ready");
  if (ready.length === 0) return null;
  return ready.reduce((best, h) => {
    if (depth(h) < depth(best)) return h;
    if (depth(h) > depth(best)) return best;
    const hFinish = h.predictedCompletionMs;
    const bestFinish = best.predictedCompletionMs;
    if (hFinish != null && bestFinish != null && hFinish !== bestFinish)
      return hFinish < bestFinish ? h : best;
    if (isOrigin(h) && !isOrigin(best)) return h;
    if (h.id < best.id) return h;
    return best;
  });
}

/** Capability ladder: CUDA (2) > Metal (1) > CPU/unknown (0). */
export function backendRank(backend: string | null): number {
  if (backend === "cuda") return 2;
  if (backend === "metal") return 1;
  return 0;
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
  let ready = hosts.filter((h) => h.status === "ready");
  if (modelHostIds !== null) {
    const withModel = ready.filter((h) => modelHostIds.includes(h.id));
    if (withModel.length > 0) ready = withModel;
  }
  if (ready.length === 0) return null;
  const rank = (x: RoutableHost) => backendRank(x.gpu?.backend ?? null);
  const vram = (x: RoutableHost) => x.gpu?.vramTotalMb ?? 0;
  return ready.reduce((best, h) => {
    if (rank(h) !== rank(best)) return rank(h) > rank(best) ? h : best;
    if (vram(h) !== vram(best)) return vram(h) > vram(best) ? h : best;
    if (depth(h) !== depth(best)) return depth(h) < depth(best) ? h : best;
    if (isOrigin(h) && !isOrigin(best)) return h;
    return best;
  });
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
  if (!selection || selection === AUTO_TARGET_ID) return AUTO_TARGET_ID;
  if (selection === CAPABLE_TARGET_ID) return CAPABLE_TARGET_ID;
  return hosts.some((h) => h.id === selection) ? selection : AUTO_TARGET_ID;
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
  return { hostId: chosen.id, label: chosen.label, target };
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
  const byName = new Map<string, ModelInfoExtended>();
  for (const id of hostIds) {
    for (const model of modelsByHost[id] ?? []) {
      const existing = byName.get(model.name);
      if (!existing || (!existing.downloaded && model.downloaded)) {
        byName.set(model.name, model);
      }
    }
  }
  return [...byName.values()];
}

/** Ids of the hosts that have `name` downloaded — the model-aware routing input. */
export function hostIdsForModel(
  modelsByHost: ModelsByHost,
  name: string,
): string[] {
  return Object.entries(modelsByHost)
    .filter(([, models]) => models.some((m) => m.name === name && m.downloaded))
    .map(([id]) => id);
}
