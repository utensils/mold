import { apiJsonTo, type ApiTarget } from "./client";

export interface ActiveWorkItem {
  id: string;
  kind: string;
  phase: string;
  model?: string | null;
  created_at_unix_ms: number;
  updated_at_unix_ms: number;
  position?: number | null;
  current?: number | null;
  total?: number | null;
  can_cancel: boolean;
}

export interface ActiveWorkSnapshot {
  instance_id: string;
  observed_at_unix_ms: number;
  items: ActiveWorkItem[];
  unavailable_kinds?: string[];
}

export interface ActivityHostRoute {
  hostId: string;
  hostLabel: string;
  target: ApiTarget;
}

export interface ActivityHostSnapshot extends ActivityHostRoute {
  /** Safe route fence persisted without the API key. */
  routeUrl: string;
  instanceId: string | null;
  observedAtUnixMs: number | null;
  items: ActiveWorkItem[];
  unavailableKinds: string[];
  stale: boolean;
  error: string | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function parseActiveWorkSnapshot(value: unknown): ActiveWorkSnapshot {
  if (
    !isRecord(value) ||
    typeof value.instance_id !== "string" ||
    value.instance_id.length === 0 ||
    !Number.isFinite(value.observed_at_unix_ms) ||
    !Array.isArray(value.items)
  ) {
    throw new Error("Host returned an incompatible active-work snapshot");
  }
  const items = value.items.map((raw, index) => {
    if (
      !isRecord(raw) ||
      typeof raw.id !== "string" ||
      typeof raw.kind !== "string" ||
      typeof raw.phase !== "string" ||
      !Number.isFinite(raw.created_at_unix_ms) ||
      !Number.isFinite(raw.updated_at_unix_ms)
    ) {
      throw new Error(
        `Host returned an incompatible active-work item at index ${index}`,
      );
    }
    return { can_cancel: false, ...raw } as ActiveWorkItem;
  });
  const unavailable = Array.isArray(value.unavailable_kinds)
    ? value.unavailable_kinds.filter(
        (kind): kind is string => typeof kind === "string",
      )
    : [];
  return {
    instance_id: value.instance_id,
    observed_at_unix_ms: value.observed_at_unix_ms as number,
    items,
    unavailable_kinds: unavailable,
  };
}

export async function listActiveWork(
  target: ApiTarget,
): Promise<ActiveWorkSnapshot> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 9_000);
  try {
    return parseActiveWorkSnapshot(
      await apiJsonTo<unknown>(target, "/api/activity", {
        signal: controller.signal,
      }),
    );
  } finally {
    clearTimeout(timeout);
  }
}

/** Replace a successful snapshot wholesale. On failure retain the last
 * verified items and mark them stale — an offline host is not evidence that
 * its work vanished. `requestEpoch` ownership lives in each surface so an
 * older HTTP response can never overwrite a newer one. */
export function reconcileActivityHost(
  route: ActivityHostRoute,
  previous: ActivityHostSnapshot | undefined,
  result: ActiveWorkSnapshot | Error,
): ActivityHostSnapshot {
  if (result instanceof Error) {
    const routeChanged =
      previous?.routeUrl != null && previous.routeUrl !== route.target.baseUrl;
    return {
      ...route,
      routeUrl: route.target.baseUrl,
      instanceId: previous?.instanceId ?? null,
      observedAtUnixMs: previous?.observedAtUnixMs ?? null,
      items: routeChanged ? [] : (previous?.items ?? []),
      unavailableKinds: previous?.unavailableKinds ?? [],
      stale: true,
      error: result.message,
    };
  }
  const unavailableKinds = result.unavailable_kinds ?? [];
  const unavailable = new Set(unavailableKinds);
  const sameAuthority =
    previous?.routeUrl === route.target.baseUrl &&
    previous.instanceId === result.instance_id;
  const retained =
    (sameAuthority
      ? previous?.items.filter((item) => unavailable.has(item.kind))
      : undefined) ?? [];
  return {
    ...route,
    routeUrl: route.target.baseUrl,
    instanceId: result.instance_id,
    observedAtUnixMs: result.observed_at_unix_ms,
    items: [
      ...result.items.filter((item) => !unavailable.has(item.kind)),
      ...retained,
    ],
    unavailableKinds,
    stale: false,
    error: null,
  };
}

export interface FleetActiveWork extends ActiveWorkItem {
  key: string;
  hostId: string;
  hostLabel: string;
  stale: boolean;
  hostError: string | null;
}

export function mergeFleetActivity(
  snapshots: Iterable<ActivityHostSnapshot>,
): FleetActiveWork[] {
  const rows: FleetActiveWork[] = [];
  for (const host of snapshots) {
    for (const item of host.items) {
      rows.push({
        ...item,
        key: `${host.hostId}:${item.kind}:${item.id}`,
        hostId: host.hostId,
        hostLabel: host.hostLabel,
        stale: host.stale || host.unavailableKinds.includes(item.kind),
        hostError: host.unavailableKinds.includes(item.kind)
          ? `${item.kind.replaceAll("_", " ")} status is temporarily unavailable`
          : host.error,
      });
    }
  }
  const phaseRank = (phase: string) =>
    phase === "running" || phase === "downloading"
      ? 0
      : phase === "loading"
        ? 1
        : 2;
  return rows.sort(
    (a, b) =>
      phaseRank(a.phase) - phaseRank(b.phase) ||
      a.created_at_unix_ms - b.created_at_unix_ms ||
      a.key.localeCompare(b.key),
  );
}
