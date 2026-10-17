import { CONNECTION_MODE_HEADER } from "@studio/api/connectionMode";
import { connectionHealth } from "@studio/api/connectionRoutes";
import {
  listStoredHosts,
  updateHost,
  ORIGIN_HOST_ID,
} from "../../lib/hostRegistry";
import { originAuthenticatedFetch as fetch } from "../../lib/originAuth";
/*
 * Per-host fetch client for the Machines workspace. Dependency-free plain
 * fetch against an arbitrary mold server's origin URL, so the same code path
 * serves the primary "this server" origin and every remote entry. Keys are
 * attached as the server's `x-api-key` header (crates/mold-server/src/auth.rs),
 * never placed in a URL.
 *
 * mold servers ship permissive CORS by default, so the browser can reach other
 * hosts directly — no proxy. All helpers are typed against src/types.ts, with a
 * current additive fields (`instance_id`, `models_disk`, `queue_paused`, queue
 * capabilities).
 */
import {
  getCurrentInstance,
  isRef,
  onBeforeUnmount,
  ref,
  watch,
  type Ref,
} from "vue";
import type {
  DownloadsListingWire,
  ModelInfoExtended,
  QueueListing,
  ResourceSnapshot,
  ServerCapabilities,
  ServerStatus,
} from "../../types";
import type { HostEntry } from "../../lib/hostRegistry";
import {
  ApiHttpError,
  looksLikeCatalogId,
  type CatalogDownloadResponse,
} from "../../api";
import {
  ApiError,
  parseCurrentServerStatus,
  type ApiTarget,
} from "@studio/api/client";
import type { GalleryView } from "@studio/lib/api/galleryOrganization";
import {
  galleryChangesFeedAdvertised,
  httpGalleryReplicaTransport,
} from "@studio/api/galleryChanges";
import { GalleryReplica } from "@studio/lib/galleryReplica";
import type { ConfigValue } from "@studio/api/config";
import {
  listDevices,
  setDeviceEnabled,
  type DeviceInfo,
  type DeviceListResponse,
} from "@studio/api/devices";
import {
  parseQueueListing,
  queueListingPath,
  queuePageRequestForCapacity,
  type QueuePageRequest,
} from "@studio/api/queuePlan";

export type HostStatus = ServerStatus;
export type HostCapabilities = ServerCapabilities;

/** Server-assisted DNS-SD result. Addresses are returned for direct browser
 * connections; the primary server does not proxy peer traffic. */
export interface DiscoveryPeer {
  name: string;
  url: string;
  host: string;
  port: number;
  version: string | null;
  auth_required: boolean;
  instance_id: string | null;
  is_this_machine: boolean;
}

function authHeaders(host: HostEntry): Record<string, string> {
  return {
    ...(host.apiKey ? { "x-api-key": host.apiKey } : {}),
    ...(host.connectionMode
      ? { [CONNECTION_MODE_HEADER]: host.connectionMode }
      : {}),
  };
}

async function getJson<T>(
  host: HostEntry,
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  const res = await fetch(`${host.url}${path}`, {
    headers: authHeaders(host),
    signal,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new ApiHttpError(`GET ${path}`, res.status, detail);
  }
  return (await res.json()) as T;
}

async function send(
  host: HostEntry,
  path: string,
  method: string,
  body?: unknown,
  allowNotFound = true,
): Promise<void> {
  const headers: Record<string, string> = { ...authHeaders(host) };
  if (body !== undefined) headers["content-type"] = "application/json";
  const res = await fetch(`${host.url}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  // DELETE routes answer 204. A missing queue id is not proof that this
  // client's job stopped, so cancellation callers must observe 404.
  if (!res.ok && res.status !== 204 && (res.status !== 404 || !allowNotFound)) {
    throw new Error(`${method} ${path} failed: ${res.status}`);
  }
}

export async function hostStatus(host: HostEntry, signal?: AbortSignal) {
  const read = async (url: string) => {
    const value = await getJson<unknown>(
      { ...host, url },
      "/api/status",
      signal,
    );
    parseCurrentServerStatus(value);
    return value as HostStatus;
  };
  if (host.id === ORIGIN_HOST_ID) return read(host.url);
  const initial = listStoredHosts().find((entry) => entry.id === host.id);
  const isCurrent = () => {
    if (!initial) return true;
    const current = listStoredHosts().find((entry) => entry.id === host.id);
    return (
      !!current &&
      (current.url === initial.url || current.url === host.url) &&
      current.apiKey === initial.apiKey &&
      current.connectionMode === initial.connectionMode &&
      current.instanceId === initial.instanceId &&
      current.connected !== false
    );
  };
  const resolved = await connectionHealth({
    hostId: host.id,
    baseUrl: host.url,
    apiKey: host.apiKey,
    connectionMode: host.connectionMode,
    instanceId: host.instanceId,
    signal,
    isCurrent,
    read,
  });
  if (initial && isCurrent() && resolved.baseUrl !== host.url) {
    updateHost(host.id, { url: resolved.baseUrl }, true);
    host.url = resolved.baseUrl;
  }
  return resolved.value;
}

/** Current servers expose the authoritative full device inventory here.
 * Callers deliberately treat a failed probe as an older-server fallback. */
export function hostDevices(
  host: HostEntry,
  signal?: AbortSignal,
): Promise<DeviceListResponse> {
  return listDevices(
    {
      baseUrl: host.url,
      apiKey: host.apiKey ?? null,
      connectionMode: host.connectionMode,
    },
    signal,
  );
}

export function setHostDeviceEnabled(
  host: HostEntry,
  deviceId: string,
  enabled: boolean,
): Promise<DeviceInfo> {
  return setDeviceEnabled(
    {
      baseUrl: host.url,
      apiKey: host.apiKey ?? null,
      connectionMode: host.connectionMode,
    },
    deviceId,
    enabled,
  );
}

export function hostDiscoveryPeers(host: HostEntry, signal?: AbortSignal) {
  return getJson<DiscoveryPeer[]>(host, "/api/discovery/peers", signal);
}

/** The studio `ApiTarget` for a web host entry: the explicit-target helpers
 * in `@studio/api/*` take this shape and send the key as `x-api-key`. */
export function hostApiTarget(host: HostEntry): ApiTarget {
  return {
    baseUrl: host.url,
    apiKey: host.apiKey ?? null,
    connectionMode: host.connectionMode,
  };
}

type WebGalleryRow = import("../../types").GalleryImage;

/** How long a host's capability answer decides feed-vs-listing. */
const REPLICA_CAPABILITY_TTL_MS = 60_000;
/** A read within this window of a finished pull reuses it: the library and
 * trash reads of one refresh are one round of requests. Change evidence
 * (an SSE tickle) pulls the replica directly and is never swallowed. */
const HOST_GALLERY_READ_FRESH_MS = 2_000;
interface ReplicaSlot {
  hostId: string;
  replica: GalleryReplica<WebGalleryRow>;
}
const galleryReplicas = new Map<string, ReplicaSlot>();

function replicaKey(host: HostEntry): string {
  return [host.id, host.url, host.apiKey ?? "", host.connectionMode ?? ""]
    .map((part) => encodeURIComponent(part))
    .join("|");
}

/**
 * The one in-memory gallery replica for this host route (id + URL + key +
 * connection mode — a re-keyed or re-pointed host gets a fresh one). It
 * follows `GET /api/gallery/changes` when the host advertises
 * `gallery.changes_feed`, else the conditional `/api/gallery` (+
 * `?view=trash` on a trash-capable host) listings, so a pull is O(delta)
 * wherever the server allows it.
 */
function hostGalleryReplica(host: HostEntry): GalleryReplica<WebGalleryRow> {
  const key = replicaKey(host);
  const existing = galleryReplicas.get(key);
  if (existing) return existing.replica;
  // A re-keyed route replaces its predecessor; a host that left the
  // registry takes its replica with it.
  const registered = new Set([
    ORIGIN_HOST_ID,
    host.id,
    ...listStoredHosts().map((stored) => stored.id),
  ]);
  for (const [other, slot] of galleryReplicas) {
    if (slot.hostId !== host.id && registered.has(slot.hostId)) continue;
    slot.replica.dispose();
    galleryReplicas.delete(other);
  }
  let capabilities: { at: number; value: Promise<HostCapabilities> } | null =
    null;
  // Only an ANSWER is cached: a failed probe is forgotten at once, so a
  // transient failure never pins the listing path for the whole TTL.
  const caps = (): Promise<HostCapabilities> => {
    const now = Date.now();
    if (!capabilities || now - capabilities.at > REPLICA_CAPABILITY_TTL_MS) {
      const entry = {
        at: now,
        value: getJson<HostCapabilities>(host, "/api/capabilities"),
      };
      capabilities = entry;
      entry.value.catch(() => {
        if (capabilities === entry) capabilities = null;
      });
    }
    return capabilities.value.catch(() => DEFAULT_CAPABILITIES);
  };
  const replica = new GalleryReplica<WebGalleryRow>(
    httpGalleryReplicaTransport<WebGalleryRow>(hostApiTarget(host), {
      feedSupported: async () => galleryChangesFeedAdvertised(await caps()),
      listTrash: async () => (await caps()).gallery?.trash?.enabled === true,
    }),
  );
  galleryReplicas.set(key, { hostId: host.id, replica });
  return replica;
}

/** The CALLER stops waiting when its signal aborts; the shared pull keeps
 * running for the replica's other readers. */
function untilAborted<T>(work: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return work;
  if (signal.aborted)
    return Promise.reject(
      signal.reason ?? new DOMException("Aborted", "AbortError"),
    );
  return new Promise<T>((resolve, reject) => {
    const abort = () =>
      reject(signal.reason ?? new DOMException("Aborted", "AbortError"));
    signal.addEventListener("abort", abort, { once: true });
    work.then(
      (value) => {
        signal.removeEventListener("abort", abort);
        resolve(value);
      },
      (error: unknown) => {
        signal.removeEventListener("abort", abort);
        reject(error);
      },
    );
  });
}

/** One host's gallery view (`library` or `trash`), read from its replica.
 * Used by the multi-host gallery merge; works for the origin too (empty
 * `host.url`). A read JOINS a pull already in flight, so the library and
 * trash reads of one refresh cost one round of requests. The returned array
 * is the replica's memoized view — the identical array across reads until
 * something changed, so a caller can skip all work on an idle pull. A trash read rejects when the host's trash was not
 * listed this pull, so callers never treat a held copy as fresh evidence. */
export async function hostGallery(
  host: HostEntry,
  signal?: AbortSignal,
  view: GalleryView = "library",
  options: { refresh?: boolean } = {},
): Promise<WebGalleryRow[]> {
  const replica = hostGalleryReplica(host);
  // `refresh` carries change evidence (a tickle, a mutation, a poll tick):
  // it always pulls, coalescing behind a pull in flight.
  if (options.refresh) await untilAborted(replica.pull(), signal);
  else if (
    replica.pulling ||
    Date.now() - replica.settledAt > HOST_GALLERY_READ_FRESH_MS
  )
    await untilAborted(replica.join(), signal);
  if (view === "trash") {
    if (!replica.trashFresh) throw new Error("Trash listing unavailable.");
    return replica.trash() as WebGalleryRow[];
  }
  return replica.live() as WebGalleryRow[];
}

/** One effective config row on a host (`GET /api/config/:key`). */
export async function hostConfigValue(
  host: HostEntry,
  key: string,
  signal?: AbortSignal,
): Promise<ConfigValue> {
  const row = await getJson<{ value?: ConfigValue }>(
    host,
    `/api/config/${encodeURIComponent(key)}`,
    signal,
  );
  return row.value ?? null;
}

/** Write one config key on a host (`PUT /api/config/:key`). A 404 is a real
 * failure here (unknown key / older host), never "already done". */
export function hostWriteConfig(
  host: HostEntry,
  key: string,
  value: ConfigValue,
): Promise<void> {
  return send(
    host,
    `/api/config/${encodeURIComponent(key)}`,
    "PUT",
    { value },
    false,
  );
}

/** Delete a print on its owning host (`DELETE /api/gallery/image/:filename`). */
export function hostDeleteGalleryImage(
  host: HostEntry,
  filename: string,
): Promise<void> {
  return send(
    host,
    `/api/gallery/image/${encodeURIComponent(filename)}`,
    "DELETE",
  );
}

export function hostResources(host: HostEntry, signal?: AbortSignal) {
  return getJson<ResourceSnapshot>(host, "/api/resources", signal);
}

export function hostQueue(
  host: HostEntry,
  signal?: AbortSignal,
  page?: QueuePageRequest,
) {
  const request =
    page === undefined
      ? getJson<unknown>(host, "/api/status", signal).then((status) =>
          queuePageRequestForCapacity(
            typeof status === "object" && status !== null
              ? (status as Record<string, unknown>).queue_capacity
              : undefined,
          ),
        )
      : Promise.resolve(page);
  return request.then((resolved) =>
    getJson<unknown>(host, queueListingPath(resolved), signal).then(
      (value) => parseQueueListing(value) as QueueListing,
    ),
  );
}

export function hostModels(host: HostEntry, signal?: AbortSignal) {
  return getJson<ModelInfoExtended[]>(host, "/api/models", signal);
}

export async function hostDownloads(
  host: HostEntry,
  signal?: AbortSignal,
): Promise<DownloadsListingWire> {
  const raw = await getJson<DownloadsListingWire>(
    host,
    "/api/downloads",
    signal,
  );
  return {
    active: raw.active ?? null,
    active_jobs: raw.active_jobs ?? [],
    queued: raw.queued ?? [],
    history: raw.history ?? [],
  };
}

/**
 * Enqueue a model download on ONE machine.
 *
 * Routed by id shape, exactly like `useDownloads().enqueue` does for the
 * origin: `/api/downloads` validates against the manifest registry and 400s on
 * a `cv:` / `hf:` id, while `/api/catalog/:id/download` owns the catalog
 * recipe + companion flow and 400s on a plain manifest name. The origin keeps
 * going through the catalog composable so the downloads centre repaints
 * immediately; this is the path for every other machine.
 *
 * It deliberately does not reuse `send()`, which treats a 404 as already-done —
 * for a download that would report a queued job that never existed.
 */
export async function hostModelDownload(
  host: HostEntry,
  id: string,
): Promise<CatalogDownloadResponse | null> {
  const catalog = looksLikeCatalogId(id);
  const path = catalog
    ? `/api/catalog/${encodeURIComponent(id)}/download`
    : "/api/downloads";
  const headers = authHeaders(host);
  if (!catalog) headers["content-type"] = "application/json";
  const res = await fetch(`${host.url}${path}`, {
    method: "POST",
    headers,
    ...(catalog ? {} : { body: JSON.stringify({ model: id }) }),
  });
  if (!res.ok) {
    const raw = await res.text().catch(() => "");
    // A license refusal answers JSON carrying the pinned terms. Parse it so
    // the shared consent dialog can render them; fall back to the raw text for
    // every other failure, which is all this ever used to carry.
    let body: unknown = null;
    try {
      body = raw ? JSON.parse(raw) : null;
    } catch {
      body = null;
    }
    const detail =
      body &&
      typeof body === "object" &&
      typeof (body as Record<string, unknown>).error === "string"
        ? ((body as Record<string, unknown>).error as string)
        : raw;
    if (body) throw new ApiError(detail, res.status, body);
    throw new ApiHttpError(`POST ${path}`, res.status, detail);
  }
  return catalog ? ((await res.json()) as CatalogDownloadResponse) : null;
}

const DEFAULT_CAPABILITIES: HostCapabilities = {
  gallery: { can_delete: true },
  queue: { can_pause: false, can_cancel_all: false, can_reorder: false },
};

/** Capabilities, falling back to a controls-off default when unreachable. */
export async function hostCapabilities(
  host: HostEntry,
  signal?: AbortSignal,
): Promise<HostCapabilities> {
  try {
    return await getJson<HostCapabilities>(host, "/api/capabilities", signal);
  } catch {
    return DEFAULT_CAPABILITIES;
  }
}

export function cancelQueueJob(host: HostEntry, id: string): Promise<void> {
  return send(
    host,
    `/api/queue/${encodeURIComponent(id)}`,
    "DELETE",
    undefined,
    false,
  );
}

export function pauseHostQueue(host: HostEntry): Promise<void> {
  return send(host, "/api/queue/pause", "POST");
}

export function resumeHostQueue(host: HostEntry): Promise<void> {
  return send(host, "/api/queue/resume", "POST");
}

export function cancelAllHostQueue(host: HostEntry): Promise<void> {
  return send(host, "/api/queue", "DELETE");
}

export function setQueueJobLane(
  host: HostEntry,
  id: string,
  targetGpu: number | null,
): Promise<void> {
  return send(host, `/api/queue/${encodeURIComponent(id)}`, "PATCH", {
    target_gpu: targetGpu,
  });
}

export function moveQueueJob(
  host: HostEntry,
  id: string,
  position: number,
): Promise<void> {
  return send(host, `/api/queue/${encodeURIComponent(id)}`, "PATCH", {
    position,
  });
}

export interface HostPoll {
  status: Ref<HostStatus | null>;
  /** Full current-server inventory, or null when the endpoint is unsupported. */
  devices: Ref<DeviceInfo[] | null>;
  deviceState: Ref<DeviceListResponse | null>;
  resources: Ref<ResourceSnapshot | null>;
  online: Ref<boolean>;
  /** Latest status read failed after this exact target was verified. */
  stale: Ref<boolean>;
  /** The exact target rejected its HTTP credential. */
  authorityRejected: Ref<boolean>;
  /** Epoch ms of the last successful status read, or null before the first. */
  lastSeen: Ref<number | null>;
  error: Ref<string | null>;
  /** Whether the first poll is still in flight (drives skeletons). */
  loading: Ref<boolean>;
  refresh: () => Promise<void>;
  stop: () => void;
}

export interface HostPollOptions {
  intervalMs?: number;
  /** Also poll `/api/resources` (host detail telemetry). */
  withResources?: boolean;
}

/**
 * Abortable status poll for one host. Sets `online` from the last status read
 * and stamps `lastSeen` on success. A same-target failure keeps last-good data
 * so the UI can dim it (spec §08 G4); changing id, URL, or key starts a clean
 * target session immediately so one machine can never display another's data.
 */
export function useHostPoll(
  host: HostEntry | Readonly<Ref<HostEntry | null>>,
  options: HostPollOptions = {},
): HostPoll {
  const intervalMs = options.intervalMs ?? 5000;
  const status = ref<HostStatus | null>(null);
  const devices = ref<DeviceInfo[] | null>(null);
  const deviceState = ref<DeviceListResponse | null>(null);
  const resources = ref<ResourceSnapshot | null>(null);
  const online = ref(false);
  const stale = ref(false);
  const authorityRejected = ref(false);
  const lastSeen = ref<number | null>(null);
  const error = ref<string | null>(null);
  const loading = ref(true);

  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let controller: AbortController | null = null;
  let refreshInFlight: Promise<void> | null = null;
  let followUpRequested = false;
  const currentHost = () => (isRef(host) ? host.value : host);
  const isCurrentTarget = (target: HostEntry) => {
    const current = currentHost();
    return (
      current?.id === target.id &&
      current.url === target.url &&
      current.apiKey === target.apiKey
    );
  };
  const clearTargetState = () => {
    status.value = null;
    devices.value = null;
    deviceState.value = null;
    resources.value = null;
    online.value = false;
    stale.value = false;
    authorityRejected.value = false;
    lastSeen.value = null;
    error.value = null;
    loading.value = true;
  };

  async function refreshOnce(): Promise<void> {
    const requestController = new AbortController();
    controller = requestController;
    const signal = requestController.signal;
    const target = currentHost();
    if (!target) {
      clearTargetState();
      loading.value = false;
      return;
    }
    try {
      const statusRequest = hostStatus(target, signal);
      const [nextStatus, resourceResult, deviceResult] = await Promise.all([
        statusRequest,
        options.withResources
          ? statusRequest
              .then(() => hostResources(target, signal))
              .then(
                (value) => ({ value, error: null }),
                (error: unknown) => ({ value: null, error }),
              )
          : Promise.resolve({ value: null, error: null }),
        statusRequest
          .then(() => hostDevices(target, signal))
          .then(
            (value) => ({ value, error: null }),
            (error: unknown) => ({ value: null, error }),
          ),
      ]);
      if (signal.aborted || !isCurrentTarget(target)) return;
      const auxiliaryAuthorityError = [
        resourceResult.error,
        deviceResult.error,
      ].find(
        (error) =>
          (error instanceof ApiHttpError || error instanceof ApiError) &&
          (error.status === 401 || error.status === 403),
      );
      if (auxiliaryAuthorityError) throw auxiliaryAuthorityError;
      status.value = nextStatus;
      if (deviceResult.error === null) {
        deviceState.value = deviceResult.value;
        devices.value = deviceResult.value?.devices ?? null;
      }
      if (options.withResources && resourceResult.error === null)
        resources.value = resourceResult.value;
      online.value = true;
      stale.value = false;
      authorityRejected.value = false;
      lastSeen.value = Date.now();
      error.value = null;
    } catch (e) {
      if (signal.aborted || !isCurrentTarget(target)) return;
      const rejected =
        (e instanceof ApiHttpError || e instanceof ApiError) &&
        (e.status === 401 || e.status === 403);
      if (rejected) {
        status.value = null;
        devices.value = null;
        deviceState.value = null;
        resources.value = null;
        online.value = false;
        stale.value = false;
        authorityRejected.value = true;
      } else {
        const verified = status.value !== null;
        online.value = verified;
        stale.value = verified;
        authorityRejected.value = false;
      }
      error.value = e instanceof Error ? e.message : String(e);
    } finally {
      if (!signal.aborted) loading.value = false;
      if (controller === requestController) controller = null;
    }
  }

  function refresh(): Promise<void> {
    if (refreshInFlight) {
      followUpRequested = true;
      return refreshInFlight;
    }
    const run = (async () => {
      do {
        followUpRequested = false;
        await refreshOnce();
      } while (!stopped && followUpRequested);
    })().finally(() => {
      if (refreshInFlight === run) refreshInFlight = null;
    });
    refreshInFlight = run;
    return run;
  }

  async function tick(): Promise<void> {
    if (stopped) return;
    await refresh();
    if (!stopped) timer = setTimeout(tick, intervalMs);
  }

  function stop(): void {
    stopped = true;
    if (timer) clearTimeout(timer);
    timer = null;
    controller?.abort();
    controller = null;
    followUpRequested = false;
  }

  if (isRef(host)) {
    watch(
      () => {
        const target = currentHost();
        return target
          ? `${target.id}\u0000${target.url}\u0000${target.apiKey ?? ""}`
          : null;
      },
      () => {
        controller?.abort();
        clearTargetState();
        void refresh();
      },
      { flush: "sync" },
    );
  }
  void tick();
  if (getCurrentInstance()) onBeforeUnmount(stop);

  return {
    status,
    devices,
    deviceState,
    resources,
    online,
    stale,
    authorityRejected,
    lastSeen,
    error,
    loading,
    refresh,
    stop,
  };
}
