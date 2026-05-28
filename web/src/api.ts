import type {
  ChainProgressEvent,
  ChainRequestWire,
  ExpandRequestWire,
  ExpandResponseWire,
  GalleryImage,
  GenerateRequestWire,
  ModelInfoExtended,
  ServerCapabilities,
  ServerStatus,
  SseChainCompleteEvent,
  SseCompleteEvent,
  SseProgressEvent,
  QueueEntry,
  QueueListing,
} from "./types";
import { streamSse } from "./lib/sse";

// Relative URLs keep the SPA portable: in dev Vite's proxy forwards to the
// mold server; in prod the SPA is served by the same server, same origin.
const base = "";

export async function listGallery(
  signal?: AbortSignal,
): Promise<GalleryImage[]> {
  const res = await fetch(`${base}/api/gallery`, { signal });
  if (!res.ok) {
    throw new Error(`GET /api/gallery failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as GalleryImage[];
}

export async function deleteGalleryImage(filename: string): Promise<void> {
  const res = await fetch(
    `${base}/api/gallery/image/${encodeURIComponent(filename)}`,
    {
      method: "DELETE",
    },
  );
  if (!res.ok && res.status !== 204) {
    throw new Error(`DELETE failed: ${res.status} ${res.statusText}`);
  }
}

/**
 * Fetch server capabilities. Delete is always enabled on current builds;
 * the capability struct is kept so older clients still see a stable shape.
 */
export async function fetchCapabilities(): Promise<ServerCapabilities> {
  try {
    const res = await fetch(`${base}/api/capabilities`);
    if (!res.ok) return defaultCapabilities();
    return (await res.json()) as ServerCapabilities;
  } catch {
    return defaultCapabilities();
  }
}

function defaultCapabilities(): ServerCapabilities {
  return { gallery: { can_delete: true } };
}

export interface ChainLimits {
  model: string;
  frames_per_clip_cap: number;
  frames_per_clip_recommended: number;
  max_stages: number;
  max_total_frames: number;
  fade_frames_max: number;
  transition_modes: string[];
  quantization_family: string;
  supports_audio: boolean;
}

export function imageUrl(filename: string): string {
  return `${base}/api/gallery/image/${encodeURIComponent(filename)}`;
}

export function thumbnailUrl(filename: string): string {
  return `${base}/api/gallery/thumbnail/${encodeURIComponent(filename)}`;
}

export async function fetchModels(
  signal?: AbortSignal,
): Promise<ModelInfoExtended[]> {
  const res = await fetch(`${base}/api/models`, { signal });
  if (!res.ok) throw new Error(`GET /api/models failed: ${res.status}`);
  return (await res.json()) as ModelInfoExtended[];
}

export async function fetchStatus(signal?: AbortSignal): Promise<ServerStatus> {
  const res = await fetch(`${base}/api/status`, { signal });
  if (!res.ok) throw new Error(`GET /api/status failed: ${res.status}`);
  return (await res.json()) as ServerStatus;
}

/** Snapshot the server's authoritative list of in-flight jobs. Used by the
 * client-side reconciliation loop to dead-letter "running" cards whose
 * `serverId` no longer appears in the registry (zombie from a dropped SSE
 * stream). Throws on transport failure so the caller can back off; the
 * empty-listing case (`{entries: []}`) is a successful response. */
export async function fetchQueue(signal?: AbortSignal): Promise<QueueListing> {
  const res = await fetch(`${base}/api/queue`, { signal });
  if (!res.ok) throw new Error(`GET /api/queue failed: ${res.status}`);
  return (await res.json()) as QueueListing;
}

export async function updateQueueJobTargetGpu(
  id: string,
  targetGpu: number | null,
  signal?: AbortSignal,
): Promise<QueueEntry> {
  const res = await fetch(`${base}/api/queue/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ target_gpu: targetGpu }),
    signal,
  });
  if (!res.ok) throw new Error(`PATCH /api/queue/${id} failed: ${res.status}`);
  return (await res.json()) as QueueEntry;
}

const chainLimitsCache = new Map<string, { value: ChainLimits; at: number }>();
const CHAIN_LIMITS_TTL_MS = 30_000;

export async function fetchChainLimits(model: string): Promise<ChainLimits> {
  const now = Date.now();
  const cached = chainLimitsCache.get(model);
  if (cached && now - cached.at < CHAIN_LIMITS_TTL_MS) return cached.value;
  const res = await fetch(
    `${base}/api/capabilities/chain-limits?model=${encodeURIComponent(model)}`,
  );
  if (!res.ok) throw new Error(`chain-limits fetch failed: ${res.status}`);
  const value: ChainLimits = await res.json();
  chainLimitsCache.set(model, { value, at: now });
  return value;
}

export async function expandPrompt(
  req: ExpandRequestWire,
  signal?: AbortSignal,
): Promise<ExpandResponseWire> {
  const res = await fetch(`${base}/api/expand`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`POST /api/expand failed: ${res.status} ${text}`);
  }
  return (await res.json()) as ExpandResponseWire;
}

export interface GenerateStreamHandlers {
  onProgress: (evt: SseProgressEvent) => void;
  onComplete: (evt: SseCompleteEvent) => void;
  onError: (
    err:
      | { kind: "http"; status: number; retryAfter?: number; body: string }
      | { kind: "network"; message: string },
  ) => void;
}

export async function generateStream(
  req: GenerateRequestWire,
  handlers: GenerateStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  // Tracks whether a terminal SSE signal (`complete`, server `error`
  // event, or HTTP-level failure) has fired. Without this guard a
  // silent mid-stream close — network blip, server restart, proxy
  // idle timeout — leaves the caller's job in `running` forever
  // because the fetch reader resolves cleanly with no event delivered.
  let terminal = false;
  try {
    const res = await streamSse({
      url: `${base}/api/generate/stream`,
      body: req,
      signal,
      onEvent: (evt) => {
        if (!evt.data) return;
        let parsed: unknown;
        try {
          parsed = JSON.parse(evt.data);
        } catch {
          return;
        }
        if (evt.event === "complete") {
          terminal = true;
          handlers.onComplete(parsed as SseCompleteEvent);
        } else if (evt.event === "error") {
          terminal = true;
          const body = evt.data;
          handlers.onError({ kind: "http", status: 0, body });
        } else {
          // Default `progress` event — the server emits many progress types
          // tagged with an internal `type` discriminator.
          handlers.onProgress(parsed as SseProgressEvent);
        }
      },
      onHttpError: (res) => {
        terminal = true;
        const retryAfterHeader = res.headers.get("Retry-After");
        const retryAfter = retryAfterHeader
          ? Number.parseFloat(retryAfterHeader)
          : undefined;
        res
          .text()
          .then((body) =>
            handlers.onError({
              kind: "http",
              status: res.status,
              retryAfter: Number.isFinite(retryAfter) ? retryAfter : undefined,
              body,
            }),
          )
          .catch(() =>
            handlers.onError({ kind: "http", status: res.status, body: "" }),
          );
      },
    });
    void res;
    if (!terminal && !signal?.aborted) {
      handlers.onError({
        kind: "network",
        message: "stream closed before completion",
      });
    }
  } catch (err) {
    if (signal?.aborted) return; // user canceled
    if (terminal) return;
    handlers.onError({
      kind: "network",
      message: err instanceof Error ? err.message : String(err),
    });
  }
}

export interface ChainStreamHandlers {
  onProgress: (evt: ChainProgressEvent) => void;
  onComplete: (evt: SseChainCompleteEvent) => void;
  onError: (
    err:
      | { kind: "http"; status: number; retryAfter?: number; body: string }
      | { kind: "network"; message: string },
  ) => void;
}

/** POST /api/generate/chain/stream — SSE stream for chained video
 * generation. Same SSE framing as `/api/generate/stream` but with a
 * `ChainRequest` body and chain-shaped progress/complete events. */
export async function generateChainStream(
  req: ChainRequestWire,
  handlers: ChainStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  // See `generateStream` for the rationale — same silent-close guard
  // applied to the chain endpoint.
  let terminal = false;
  try {
    const res = await streamSse({
      url: `${base}/api/generate/chain/stream`,
      body: req,
      signal,
      onEvent: (evt) => {
        if (!evt.data) return;
        let parsed: unknown;
        try {
          parsed = JSON.parse(evt.data);
        } catch {
          return;
        }
        if (evt.event === "complete") {
          terminal = true;
          handlers.onComplete(parsed as SseChainCompleteEvent);
        } else if (evt.event === "error") {
          terminal = true;
          handlers.onError({ kind: "http", status: 0, body: evt.data });
        } else {
          handlers.onProgress(parsed as ChainProgressEvent);
        }
      },
      onHttpError: (res) => {
        terminal = true;
        const retryAfterHeader = res.headers.get("Retry-After");
        const retryAfter = retryAfterHeader
          ? Number.parseFloat(retryAfterHeader)
          : undefined;
        res
          .text()
          .then((body) =>
            handlers.onError({
              kind: "http",
              status: res.status,
              retryAfter: Number.isFinite(retryAfter) ? retryAfter : undefined,
              body,
            }),
          )
          .catch(() =>
            handlers.onError({ kind: "http", status: res.status, body: "" }),
          );
      },
    });
    void res;
    if (!terminal && !signal?.aborted) {
      handlers.onError({
        kind: "network",
        message: "stream closed before completion",
      });
    }
  } catch (err) {
    if (signal?.aborted) return;
    if (terminal) return;
    handlers.onError({
      kind: "network",
      message: err instanceof Error ? err.message : String(err),
    });
  }
}

// ─── Downloads UI (Agent A) ───────────────────────────────────────────────────
import type { DownloadJobWire, DownloadsListingWire } from "./types";

export interface CreateDownloadResponse {
  id: string;
  position: number;
}

export async function postDownload(
  model: string,
  signal?: AbortSignal,
): Promise<{ status: "created" | "duplicate"; id: string; position: number }> {
  const res = await fetch(`${base}/api/downloads`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
    signal,
  });
  if (res.status === 400) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `unknown model '${model}'`);
  }
  if (!res.ok && res.status !== 409) {
    throw new Error(`POST /api/downloads failed: ${res.status}`);
  }
  const json = (await res.json()) as CreateDownloadResponse;
  return {
    status: res.status === 409 ? "duplicate" : "created",
    id: json.id,
    position: json.position,
  };
}

export async function cancelDownload(id: string): Promise<void> {
  const res = await fetch(`${base}/api/downloads/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (res.status === 404) {
    // Idempotent — treat as already gone.
    return;
  }
  if (!res.ok && res.status !== 204) {
    throw new Error(`DELETE /api/downloads failed: ${res.status}`);
  }
}

export async function fetchDownloads(
  signal?: AbortSignal,
): Promise<DownloadsListingWire> {
  const res = await fetch(`${base}/api/downloads`, { signal });
  if (!res.ok) throw new Error(`GET /api/downloads failed: ${res.status}`);
  const raw = (await res.json()) as DownloadsListingWire;
  // Server may omit `active`/`history` as null; normalise.
  return {
    active: raw.active ?? null,
    queued: raw.queued ?? [],
    history: raw.history ?? [],
  };
}

/** Returns the absolute URL for the SSE stream (consumed via EventSource). */
export function downloadsStreamUrl(): string {
  return `${base}/api/downloads/stream`;
}

export type { DownloadJobWire, DownloadsListingWire };
// ── Resource telemetry (Agent B) ─────────────────────────────────────────────
import type { ResourceSnapshot } from "./types";

export async function fetchResources(
  signal?: AbortSignal,
): Promise<ResourceSnapshot> {
  const res = await fetch("/api/resources", { signal });
  if (!res.ok) throw new Error(`fetchResources failed: ${res.status}`);
  return (await res.json()) as ResourceSnapshot;
}

// ─── Catalog (live HF + Civitai) ──────────────────────────────────────────
import type {
  CatalogEntryWire,
  CatalogFamiliesResponse,
  CatalogListParams,
  CatalogListResponse,
} from "./types";

/// Live HF + Civitai search proxied through the server with a 5-min cache.
export async function fetchCatalogSearch(
  params: CatalogListParams,
): Promise<CatalogListResponse> {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null) continue;
    sp.set(k, String(v));
  }
  const r = await fetch(`/api/catalog/search?${sp.toString()}`);
  if (!r.ok) throw new Error(`/api/catalog/search ${r.status}`);
  return r.json();
}

/// Enumerate installed catalog entries from per-install sidecar files.
/// Used by the LoRA picker (and any future "show what's downloaded"
/// surfaces). Family/kind are server-side filters.
export async function fetchCatalogInstalled(params: {
  family?: string;
  kind?: string;
}): Promise<CatalogListResponse> {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null) continue;
    sp.set(k, String(v));
  }
  const r = await fetch(`/api/catalog/installed?${sp.toString()}`);
  if (!r.ok) throw new Error(`/api/catalog/installed ${r.status}`);
  return r.json();
}

export async function fetchCatalogEntry(id: string): Promise<CatalogEntryWire> {
  const r = await fetch(`/api/catalog/${encodeURIComponent(id)}`);
  if (!r.ok) throw new Error(`/api/catalog/${id} ${r.status}`);
  return r.json();
}

export async function fetchCatalogFamilies(): Promise<CatalogFamiliesResponse> {
  const r = await fetch(`/api/catalog/families`);
  if (!r.ok) throw new Error(`/api/catalog/families ${r.status}`);
  return r.json();
}

export interface CompanionJob {
  name: string;
  job_id: string;
}

export interface CatalogDownloadResponse {
  /// Queue id for the catalog entry itself. `null` for Civitai entries
  /// whose recipe-driven download path is implemented in 2.8 — companion
  /// downloads (CLIP-L / CLIP-G / VAE) still flow through `companion_jobs`.
  primary_job_id: string | null;
  /// One entry per canonical companion the server enqueued. Present in
  /// the same order the catalog row's `companions` field declares.
  companion_jobs: CompanionJob[];
}

export async function postCatalogDownload(
  id: string,
): Promise<CatalogDownloadResponse> {
  const r = await fetch(`/api/catalog/${encodeURIComponent(id)}/download`, {
    method: "POST",
  });
  if (!r.ok) throw new Error(`/api/catalog/${id}/download ${r.status}`);
  return r.json();
}

/// True for values that have the structural shape of a catalog id
/// (`cv:<civitai-version-id>` or `hf:<author>/<name>`). Mirrors the
/// server-side check in `crates/mold-cli/src/catalog_bridge.rs`.
///
/// Manifest names like `flux-dev:q4` do NOT match — the colon there
/// separates `model` from `tag`, and the head never starts with `cv`/`hf`.
export function looksLikeCatalogId(input: string): boolean {
  return input.startsWith("cv:") || input.startsWith("hf:");
}
