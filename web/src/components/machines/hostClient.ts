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
  conditionalApiJsonTo,
  ApiError,
  parseCurrentServerStatus,
  type ApiTarget,
} from "@studio/api/client";
import type { GalleryView } from "@studio/lib/api/galleryOrganization";
import type { ConfigValue } from "../../lib/settingsConfig";
import {
  listDevices,
  setDeviceEnabled,
  type DeviceInfo,
  type DeviceListResponse,
} from "@studio/api/devices";
import {
  parseQueueListing,
  queueListingPath,
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

function authHeaders(key?: string): Record<string, string> {
  return key ? { "x-api-key": key } : {};
}

async function getJson<T>(
  host: HostEntry,
  path: string,
  signal?: AbortSignal,
): Promise<T> {
  const res = await fetch(`${host.url}${path}`, {
    headers: authHeaders(host.apiKey),
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
  const headers: Record<string, string> = { ...authHeaders(host.apiKey) };
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
  const value = await getJson<unknown>(host, "/api/status", signal);
  parseCurrentServerStatus(value);
  return value as HostStatus;
}

/** Current servers expose the authoritative full device inventory here.
 * Callers deliberately treat a failed probe as an older-server fallback. */
export function hostDevices(
  host: HostEntry,
  signal?: AbortSignal,
): Promise<DeviceListResponse> {
  return listDevices(
    { baseUrl: host.url, apiKey: host.apiKey ?? null },
    signal,
  );
}

export function setHostDeviceEnabled(
  host: HostEntry,
  deviceId: string,
  enabled: boolean,
): Promise<DeviceInfo> {
  return setDeviceEnabled(
    { baseUrl: host.url, apiKey: host.apiKey ?? null },
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
  return { baseUrl: host.url, apiKey: host.apiKey ?? null };
}

/** One host's gallery (`GET /api/gallery[?view=trash]`), with its
 * `x-api-key`. Used by the multi-host gallery merge. Works for the origin
 * too (empty `host.url`). The default `library` view keeps the bare path so
 * older hosts, which know no `view` parameter, behave exactly as before. */
export function hostGallery(
  host: HostEntry,
  signal?: AbortSignal,
  view: GalleryView = "library",
) {
  const path =
    view === "library" ? "/api/gallery" : `/api/gallery?view=${view}`;
  return conditionalApiJsonTo<import("../../types").GalleryImage[]>(
    hostApiTarget(host),
    path,
    { signal },
  );
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
  return getJson<unknown>(host, queueListingPath(page), signal).then(
    (value) => parseQueueListing(value) as QueueListing,
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
  const headers = authHeaders(host.apiKey);
  if (!catalog) headers["content-type"] = "application/json";
  const res = await fetch(`${host.url}${path}`, {
    method: "POST",
    headers,
    ...(catalog ? {} : { body: JSON.stringify({ model: id }) }),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
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
      const [nextStatus, resourceResult, deviceResult] = await Promise.all([
        hostStatus(target, signal),
        options.withResources
          ? hostResources(target, signal).then(
              (value) => ({ value, error: null }),
              (error: unknown) => ({ value: null, error }),
            )
          : Promise.resolve({ value: null, error: null }),
        hostDevices(target, signal).then(
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
