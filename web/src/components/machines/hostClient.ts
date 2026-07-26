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
import { onBeforeUnmount, ref, type Ref } from "vue";
import type {
  DownloadsListingWire,
  ModelInfoExtended,
  QueueListing,
  ResourceSnapshot,
  ServerCapabilities,
  ServerStatus,
} from "../../types";
import type { HostEntry } from "../../lib/hostRegistry";
import { parseCurrentServerStatus } from "@studio/api/client";
import {
  listDevices,
  type DeviceInfo,
  type DeviceListResponse,
} from "@studio/api/devices";

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
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return (await res.json()) as T;
}

async function send(
  host: HostEntry,
  path: string,
  method: string,
  body?: unknown,
): Promise<void> {
  const headers: Record<string, string> = { ...authHeaders(host.apiKey) };
  if (body !== undefined) headers["content-type"] = "application/json";
  const res = await fetch(`${host.url}${path}`, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  // DELETE routes answer 204; treat a 404 cancel as already-gone.
  if (!res.ok && res.status !== 204 && res.status !== 404) {
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
export function hostDevices(host: HostEntry): Promise<DeviceListResponse> {
  return listDevices({ baseUrl: host.url, apiKey: host.apiKey ?? null });
}

export function hostDiscoveryPeers(host: HostEntry, signal?: AbortSignal) {
  return getJson<DiscoveryPeer[]>(host, "/api/discovery/peers", signal);
}

/** One host's gallery (`GET /api/gallery`), with its `x-api-key`. Used by the
 * multi-host gallery merge. Works for the origin too (empty `host.url`). */
export function hostGallery(host: HostEntry, signal?: AbortSignal) {
  return getJson<import("../../types").GalleryImage[]>(
    host,
    "/api/gallery",
    signal,
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

export function hostQueue(host: HostEntry, signal?: AbortSignal) {
  return getJson<QueueListing>(host, "/api/queue", signal);
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
  return send(host, `/api/queue/${encodeURIComponent(id)}`, "DELETE");
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
  resources: Ref<ResourceSnapshot | null>;
  online: Ref<boolean>;
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
 * and stamps `lastSeen` on success. Offline hosts keep their last-good data so
 * the UI can dim it rather than blank out (spec §08 G4).
 */
export function useHostPoll(
  host: HostEntry,
  options: HostPollOptions = {},
): HostPoll {
  const intervalMs = options.intervalMs ?? 5000;
  const status = ref<HostStatus | null>(null);
  const devices = ref<DeviceInfo[] | null>(null);
  const resources = ref<ResourceSnapshot | null>(null);
  const online = ref(false);
  const lastSeen = ref<number | null>(null);
  const error = ref<string | null>(null);
  const loading = ref(true);

  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let controller: AbortController | null = null;

  async function refresh(): Promise<void> {
    controller?.abort();
    controller = new AbortController();
    const signal = controller.signal;
    try {
      const [nextStatus, nextResources, nextDevices] = await Promise.all([
        hostStatus(host, signal),
        options.withResources
          ? hostResources(host, signal).catch(() => null)
          : Promise.resolve(null),
        hostDevices(host).then(
          (snapshot) => snapshot.devices,
          () => null,
        ),
      ]);
      status.value = nextStatus;
      devices.value = nextDevices;
      if (options.withResources && nextResources)
        resources.value = nextResources;
      online.value = true;
      lastSeen.value = Date.now();
      error.value = null;
    } catch (e) {
      if (signal.aborted) return;
      online.value = false;
      error.value = e instanceof Error ? e.message : String(e);
    } finally {
      if (!signal.aborted) loading.value = false;
    }
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
  }

  void tick();
  onBeforeUnmount(stop);

  return {
    status,
    devices,
    resources,
    online,
    lastSeen,
    error,
    loading,
    refresh,
    stop,
  };
}
