/**
 * One generation-routing policy for every Mold Studio surface.
 *
 * Desktop, web, and iPhone all answer the same two questions — "which machine
 * should this print land on?" (Auto: model-aware least-busy) and "which machine
 * is strongest?" (Most capable: CUDA > Metal > unknown, then VRAM, then queue).
 * Those rules used to live twice, in `desktop/src/lib/hosts.ts` and
 * `web/src/lib/hostRouting.ts`, and iPhone had neither. They live here now; the
 * surface modules re-export so their callers keep their existing names.
 *
 * Surfaces differ only in their *home* host — desktop's built-in engine, web's
 * origin server — which wins a dead heat because dispatching there costs no
 * network hop. That is injected as a predicate rather than baked in; iPhone is
 * remote-only and has no home host at all.
 *
 * This module is deliberately free of HTTP imports: the placement comparator is
 * injected into `chooseRoutedHost` so routing policy stays pure and testable.
 */

/** Least-busy routing across every ready host. */
export const AUTO_TARGET_ID = "auto";
/** Strongest-GPU routing (desktop's `generateTargetHost = "capable"`). */
export const CAPABLE_TARGET_ID = "capable";

export type HostRoutingStatus = "connecting" | "ready" | "error";

/** GPU summary as the current `/api/status` contract reports it. */
export interface RoutingGpu {
  /** "cuda" | "metal"; absent from servers ≤ 0.16 — inferred from `name` then. */
  backend?: string | null;
  name?: string | null;
  vramTotalMb?: number | null;
}

/** The slice of a host the Auto router reasons over. */
export interface RoutableHostBase {
  id: string;
  status: HostRoutingStatus;
  /** Live queue depth; null while unknown (counts as busiest). */
  queueDepth: number | null;
  /** Predicted end of this host's current plan. Null on legacy hosts. */
  predictedCompletionMs?: number | null;
}

/** The slice of a host the "Most capable" router reasons over. */
export interface CapableHostBase {
  id: string;
  status: HostRoutingStatus;
  queueDepth: number | null;
  gpu: RoutingGpu | null;
}

/**
 * Surface-specific dead-heat rules. `isHome` marks the machine that costs no
 * network hop (desktop's local engine, web's origin); `lowestIdWins` keeps a
 * registry of interchangeable remotes deterministic.
 */
export interface RoutingTieBreak<T> {
  isHome?: (host: T) => boolean;
  lowestIdWins?: boolean;
}

function queueDepthOf(host: { queueDepth: number | null }): number {
  return host.queueDepth ?? Number.MAX_SAFE_INTEGER;
}

/**
 * Auto routing: when both hosts expose an authoritative plan, predicted
 * completion wins before raw queue depth. If either host is planless, queue
 * depth is the deterministic backward-compatible fallback.
 */
export function pickAutoHost<T extends RoutableHostBase>(
  hosts: readonly T[],
  tieBreak: RoutingTieBreak<T> = {},
): T | null {
  const ready = hosts.filter((host) => host.status === "ready");
  if (ready.length === 0) return null;
  return ready.reduce((best, host) => {
    const hostFinish = host.predictedCompletionMs;
    const bestFinish = best.predictedCompletionMs;
    if (hostFinish != null && bestFinish != null && hostFinish !== bestFinish)
      return hostFinish < bestFinish ? host : best;
    if (queueDepthOf(host) < queueDepthOf(best)) return host;
    if (queueDepthOf(host) > queueDepthOf(best)) return best;
    if (hostFinish != null && bestFinish == null) return host;
    if (hostFinish == null && bestFinish != null) return best;
    if (tieBreak.isHome?.(host) && !tieBreak.isHome(best)) return host;
    if (tieBreak.lowestIdWins && host.id < best.id) return host;
    return best;
  });
}

/**
 * Guess the compute backend from a GPU's marketing name. Only used when a host
 * doesn't report `gpu_info.backend` (servers ≤ 0.16) — the explicit field
 * always wins.
 */
export function inferBackendFromGpuName(name: string): "cuda" | "metal" | "cpu" {
  const lowered = name.toLowerCase();
  if (/nvidia|rtx|geforce|gtx|quadro|tesla|a100|h100|l40/.test(lowered)) return "cuda";
  if (/apple|\bm[1-4]\b/.test(lowered)) return "metal";
  return "cpu";
}

/**
 * Capability ladder: CUDA (2) > Metal (1) > CPU/unknown (0). Falls back to name
 * inference when the wire backend is missing.
 */
export function backendRank(
  backend: string | null | undefined,
  gpuName?: string | null,
): number {
  const resolved = backend ?? (gpuName ? inferBackendFromGpuName(gpuName) : null);
  if (resolved === "cuda") return 2;
  if (resolved === "metal") return 1;
  return 0;
}

/**
 * "Most capable" routing: among ready hosts — restricted to the ones that
 * already hold the model when `modelHostIds` says at least one ready host does
 * — rank by backend, then total VRAM descending (null = 0), then queue depth
 * ascending (null = busiest). A fully tied home host wins. Null when nothing is
 * ready.
 */
export function pickMostCapableHost<T extends CapableHostBase>(
  hosts: readonly T[],
  modelHostIds: readonly string[] | null,
  tieBreak: RoutingTieBreak<T> = {},
): T | null {
  let ready = hosts.filter((host) => host.status === "ready");
  if (modelHostIds !== null) {
    const withModel = ready.filter((host) => modelHostIds.includes(host.id));
    if (withModel.length > 0) ready = withModel;
  }
  if (ready.length === 0) return null;
  const rank = (host: CapableHostBase) => backendRank(host.gpu?.backend, host.gpu?.name);
  const vram = (host: CapableHostBase) => host.gpu?.vramTotalMb ?? 0;
  return ready.reduce((best, host) => {
    if (rank(host) !== rank(best)) return rank(host) > rank(best) ? host : best;
    if (vram(host) !== vram(best)) return vram(host) > vram(best) ? host : best;
    if (queueDepthOf(host) !== queueDepthOf(best))
      return queueDepthOf(host) < queueDepthOf(best) ? host : best;
    if (tieBreak.isHome?.(host) && !tieBreak.isHome(best)) return host;
    if (tieBreak.lowestIdWins && host.id < best.id) return host;
    return best;
  });
}

/**
 * Normalize a persisted target the way the picker displays it: both sentinels
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
  return hosts.some((host) => host.id === selection) ? selection : AUTO_TARGET_ID;
}

/**
 * The desktop flavour of the same normalization, where Auto is `null` rather
 * than a sentinel string (`appPrefs.settings.generateTargetHost`).
 */
export function normalizeTargetHost(
  selection: string | null | undefined,
  hosts: ReadonlyArray<{ id: string }>,
): string | null {
  const normalized = normalizeTargetId(selection, hosts);
  return normalized === AUTO_TARGET_ID ? null : normalized;
}

/** True for the two automatic policies; a concrete host id is explicit. */
export function isAutomaticTarget(selection: string | null | undefined): boolean {
  return !selection || selection === AUTO_TARGET_ID || selection === CAPABLE_TARGET_ID;
}

/** Minimal model row shape the fleet-union helpers reason over. */
export interface RoutableModel {
  name: string;
  downloaded?: boolean;
}

/**
 * The model list for a set of hosts, deduped by name. A downloaded copy always
 * beats an undownloaded one — a model installed on any host in the set is
 * installed as far as routing is concerned.
 */
export function unionModelsByName<T extends RoutableModel>(
  modelsByHost: Readonly<Record<string, readonly T[]>>,
  hostIds: readonly string[],
): T[] {
  const byName = new Map<string, T>();
  for (const id of hostIds) {
    for (const model of modelsByHost[id] ?? []) {
      const existing = byName.get(model.name);
      if (!existing || (!existing.downloaded && model.downloaded)) byName.set(model.name, model);
    }
  }
  return [...byName.values()];
}

/** Ids of the hosts that have `name` downloaded — the model-aware routing input. */
export function hostIdsForModel<T extends RoutableModel>(
  modelsByHost: Readonly<Record<string, readonly T[]>>,
  name: string,
  hostIds?: readonly string[],
): string[] {
  return Object.entries(modelsByHost)
    .filter(([id]) => !hostIds || hostIds.includes(id))
    .filter(([, models]) => models.some((model) => model.name === name && model.downloaded))
    .map(([id]) => id);
}

/** One host's answer to a placement fan-out. */
export interface PlacementFanoutEntry<T, P> {
  host: T;
  roundTripMs: number;
  preview: P;
}

/**
 * Choose the destination among the hosts that answered `planned`.
 *
 * Auto takes the shortest predicted completion including the probe's own round
 * trip (`comparePlacementPreviews`, injected so this module keeps no HTTP
 * dependency); Most capable takes the strongest GPU among the planners. Both
 * only ever see hosts that already proved they can run the request.
 */
export function chooseRoutedHost<T extends CapableHostBase, P>(
  planned: ReadonlyArray<PlacementFanoutEntry<T, P>>,
  policy: string,
  compare: (
    left: { hostId: string; roundTripMs: number; preview: P },
    right: { hostId: string; roundTripMs: number; preview: P },
  ) => number,
  tieBreak: RoutingTieBreak<T> = {},
): T | null {
  if (planned.length === 0) return null;
  if (policy === CAPABLE_TARGET_ID) {
    return pickMostCapableHost(
      planned.map((entry) => entry.host),
      null,
      tieBreak,
    );
  }
  const sorted = [...planned].sort((left, right) =>
    compare(
      { hostId: left.host.id, roundTripMs: left.roundTripMs, preview: left.preview },
      { hostId: right.host.id, roundTripMs: right.roundTripMs, preview: right.preview },
    ),
  );
  return sorted[0]?.host ?? null;
}
