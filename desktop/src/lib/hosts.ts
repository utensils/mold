/**
 * Pure multi-host helpers. `hostIdFromUrl` and `normalizeHostUrl` are TS
 * twins of `connection.rs::host_id` / `normalize_host_url` — per-host secret
 * names (`remote-api-key.<id>`) are derived on both sides and must agree.
 *
 * The routing policy itself (Auto, Most capable, the backend ladder, target
 * normalization) lives in `@studio/lib/hostRouting` so desktop, web, and the
 * iPhone app cannot drift; the wrappers below only bind desktop's home-host
 * rule — the built-in engine wins a dead heat because it costs no network hop.
 */
import {
  backendRank as sharedBackendRank,
  inferBackendFromGpuName as sharedInferBackendFromGpuName,
  normalizeTargetHost as sharedNormalizeTargetHost,
  pickAutoHost as sharedPickAutoHost,
  pickMostCapableHost as sharedPickMostCapableHost,
} from "@studio/lib/hostRouting";

export function hostIdFromUrl(url: string): string {
  const stripped = url.replace(/^https?:\/\//, "");
  return stripped
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/**
 * Display name for a host: a user-assigned name always wins; otherwise the
 * server-reported hostname (stable regardless of whether we reached the box by
 * hostname, mDNS `.local`, or IP); otherwise the URL with its scheme stripped.
 */
export function resolveHostDisplayName(
  saved: { name?: string | null; url: string } | null,
  telemetryHostname: string | null | undefined,
): string {
  if (saved?.name) return saved.name;
  if (telemetryHostname) return telemetryHostname;
  return saved?.url.replace(/^https?:\/\//, "") ?? "";
}

/** The shape the saved-host merge reasons over (a superset of `SavedHost`). */
export interface SavedHostLike {
  id: string;
  name?: string | null;
  url: string;
  lastUsedMs?: number | null;
  instanceId?: string | null;
}

/**
 * Collapse saved hosts that resolve to the same physical server (same
 * `instanceId`) — the same box reached by hostname, mDNS name, and IP produces
 * three slugs otherwise. The survivor is the entry with the most recent
 * `lastUsedMs` (ties broken by input order), keeping its slug (and thus its
 * `remote-api-key.<id>` secret and routing keys); a user-assigned name on any
 * duplicate is preserved. Entries with no `instanceId` are never merged, and
 * a shared non-empty `instanceId` is authoritative regardless of which IP,
 * hostname, or MagicDNS alias answered. Returns the deduped list
 * (input order, losers removed) and the loser→survivor id pairs so callers
 * can re-home secrets, connected ids, and a sticky generation target.
 */
export function mergeSavedHostsByInstanceId<T extends SavedHostLike>(
  hosts: T[],
): { hosts: T[]; dropped: Array<{ loser: string; survivor: string }> } {
  const groups = new Map<string, T[]>();
  for (const host of hosts) {
    if (!host.instanceId) continue;
    const group = groups.get(host.instanceId);
    if (group) group.push(host);
    else groups.set(host.instanceId, [host]);
  }
  const survivorByInstance = new Map<string, T>();
  for (const [uuid, group] of groups) {
    let survivor = group[0]!;
    for (const host of group) {
      if ((host.lastUsedMs ?? 0) > (survivor.lastUsedMs ?? 0)) survivor = host;
    }
    survivorByInstance.set(uuid, survivor);
  }
  const dropped: Array<{ loser: string; survivor: string }> = [];
  const out: T[] = [];
  for (const host of hosts) {
    const uuid = host.instanceId;
    const survivor = uuid ? survivorByInstance.get(uuid) : undefined;
    if (!survivor || survivor.id === host.id) {
      // The survivor inherits a user name from any duplicate it lacked.
      if (survivor && !survivor.name) {
        const named = hosts.find((h) => h.instanceId === uuid && h.name);
        out.push(named ? { ...survivor, name: named.name } : survivor);
      } else {
        out.push(host);
      }
      continue;
    }
    dropped.push({ loser: host.id, survivor: survivor.id });
  }
  return { hosts: out, dropped };
}

/** Default scheme/port, strip trailing slashes. Throws on garbage input. */
export function normalizeHostUrl(input: string): string {
  const trimmed = input.trim().replace(/\/+$/, "");
  if (!trimmed) throw new Error("Enter a host, like hal9000");
  const hasScheme = trimmed.startsWith("http://") || trimmed.startsWith("https://");
  const withScheme = hasScheme ? trimmed : `http://${trimmed}`;
  const url = new URL(withScheme);
  if (!url.hostname) throw new Error("Enter a valid host.");
  if (!hasScheme && !url.port) url.port = "7680";
  return url.toString().replace(/\/+$/, "");
}

/** The slice of a host the Auto router needs. */
export interface RoutableHost {
  id: string;
  kind: "local" | "remote";
  status: "connecting" | "ready" | "error";
  /** Live queue depth; null while unknown. */
  queueDepth: number | null;
  /** Latest projected completion. Unknown sorts behind a known prediction. */
  predictedCompletionMs?: number | null;
}

/**
 * Auto routing: when both hosts expose an authoritative plan, predicted
 * completion wins before raw queue depth. If either is planless, queue depth
 * remains the deterministic backward-compatible fallback.
 */
export function pickAutoHost<T extends RoutableHost>(hosts: T[]): T | null {
  return sharedPickAutoHost(hosts, { isHome: (host) => host.kind === "local" });
}

/**
 * Guess the compute backend from a GPU's marketing name. Only used when a
 * host doesn't report `gpu_info.backend` (servers ≤ 0.16) — the explicit
 * field always wins.
 */
export function inferBackendFromGpuName(name: string): "cuda" | "metal" | "cpu" {
  return sharedInferBackendFromGpuName(name);
}

/**
 * Capability ladder: CUDA (2) > Metal (1) > CPU/unknown (0). Falls back to
 * name inference when the wire backend is missing.
 */
export function backendRank(backend: string | null | undefined, gpuName?: string | null): number {
  return sharedBackendRank(backend, gpuName);
}

/** The slice of a host the "Most capable" router needs. */
export interface CapableHost {
  id: string;
  kind: "local" | "remote";
  status: "connecting" | "ready" | "error";
  /** Live queue depth; null while unknown (counts as busiest). */
  queueDepth: number | null;
  /** GPU summary from `/api/status`; null when the host reports none. */
  gpu: { backend?: string | null; name?: string | null; vramTotalMb?: number | null } | null;
}

/**
 * "Most capable" routing: among ready hosts (restricted to the ones that
 * already have the model when `modelHostIds` says at least one ready host
 * does), rank by backend (CUDA > Metal > unknown), then total VRAM
 * descending (null = 0), then queue depth ascending (null = busiest). A
 * fully tied local host wins (no network hop); everything else is
 * deterministic in favor of the stronger remote. Null when nothing is ready.
 */
export function pickMostCapableHost<T extends CapableHost>(
  hosts: T[],
  modelHostIds: string[] | null,
): T | null {
  return sharedPickMostCapableHost(hosts, modelHostIds, {
    isHome: (host) => host.kind === "local",
  });
}

/**
 * Normalize the persisted `generateTargetHost` pref the way the Host selector
 * displays it: `"capable"` and a currently-listed host id pass through; a
 * ghost id (the host was forgotten or never reconnected) reads as Auto
 * (null). Every consumer of the pref must read it through this so tag
 * suppression and picker filtering can't disagree with the selector.
 */
export function normalizeTargetHost(
  sel: string | null | undefined,
  hosts: ReadonlyArray<{ id: string }>,
): string | null {
  return sharedNormalizeTargetHost(sel, hosts);
}

/**
 * The model picker's availability tag. Quiet when availability is unknown or
 * the model is on the primary host (the default place a job lands); the
 * host's label when exactly one connected non-primary host has it; a count
 * otherwise.
 */
export function modelAvailabilityTag(
  hostIds: readonly string[],
  hosts: ReadonlyArray<{ id: string; label: string; primary: boolean }>,
): string | null {
  if (hostIds.length === 0) return null;
  const primary = hosts.find((h) => h.primary);
  if (primary && hostIds.includes(primary.id)) return null;
  const known = hosts.filter((h) => hostIds.includes(h.id));
  if (known.length === 0) return null;
  if (known.length === 1) return known[0]!.label;
  return `${known.length} hosts`;
}

/**
 * Which host the status bar should mirror: the host of the most recently
 * submitted still-live job with a routed host id when Create is using Auto or
 * Most capable, else the primary. A concrete sticky selection is resolved by
 * the status component before this fallback. Jobs without a routed host run on
 * the primary.
 */
export function pickDisplayHost(
  liveJobHostIds: ReadonlyArray<string | null>,
  primaryId: string,
): string {
  for (let i = liveJobHostIds.length - 1; i >= 0; i--) {
    const id = liveJobHostIds[i];
    if (id) return id;
  }
  return primaryId;
}

/**
 * Stable signature of the ready host set — watch this to refetch merged,
 * cross-host data (prompt history, model unions) exactly when a host comes
 * up or drops, and never on unrelated telemetry churn.
 */
export function readyHostSignature(hosts: readonly { id: string; status: string }[]): string {
  return hosts
    .filter((h) => h.status === "ready")
    .map((h) => h.id)
    .sort()
    .join(",");
}
