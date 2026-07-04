import type {
  ChainJobDetail,
  ChainJobListing,
  ChainJobSummary,
  ChainProgressEvent,
  ChainRequestWire,
  CreateChainJobResponse,
  ExpandRequestWire,
  ExpandResponseWire,
  GalleryImage,
  GenerationMemoryEstimate,
  GenerateRequestWire,
  ModelInfoExtended,
  ModelComponentsResponse,
  ServerCapabilities,
  ServerStatus,
  SseChainCompleteEvent,
  SseCompleteEvent,
  SseProgressEvent,
  SseUpscaleCompleteEvent,
  RetakeRequest,
  UpscaleRequestWire,
  QueueEntry,
  QueueListing,
} from "./types";
import { postSseJsonStream, type StreamError } from "./lib/apiStream";

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

export async function fetchGenerationEstimate(
  req: GenerateRequestWire,
  signal?: AbortSignal,
): Promise<GenerationMemoryEstimate> {
  const res = await fetch(`${base}/api/generate/estimate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok)
    throw new Error(`POST /api/generate/estimate failed: ${res.status}`);
  return (await res.json()) as GenerationMemoryEstimate;
}

export async function fetchModelComponents(
  model: string,
  signal?: AbortSignal,
): Promise<ModelComponentsResponse> {
  const res = await fetch(
    `${base}/api/models/${encodeURIComponent(model)}/components`,
    { signal },
  );
  if (!res.ok)
    throw new Error(
      `GET /api/models/${model}/components failed: ${res.status}`,
    );
  return (await res.json()) as ModelComponentsResponse;
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
  onError: (err: StreamError) => void;
}

export async function generateStream(
  req: GenerateRequestWire,
  handlers: GenerateStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  await postSseJsonStream<
    GenerateRequestWire,
    SseProgressEvent,
    SseCompleteEvent
  >({
    url: `${base}/api/generate/stream`,
    body: req,
    signal,
    handlers,
    silentCloseMessage: "stream closed before completion",
  });
}

export interface UpscaleStreamHandlers {
  onProgress: (evt: SseProgressEvent) => void;
  onComplete: (evt: SseUpscaleCompleteEvent) => void;
  onError: (err: StreamError) => void;
}

export async function upscaleStream(
  req: UpscaleRequestWire,
  handlers: UpscaleStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  await postSseJsonStream<
    UpscaleRequestWire,
    SseProgressEvent,
    SseUpscaleCompleteEvent
  >({
    url: `${base}/api/upscale/stream`,
    body: req,
    signal,
    handlers,
    silentCloseMessage: "upscale stream closed before completion",
  });
}

export interface ChainStreamHandlers {
  onProgress: (evt: ChainProgressEvent) => void;
  onComplete: (evt: SseChainCompleteEvent) => void;
  onError: (err: StreamError) => void;
}

/** POST /api/generate/chain/stream — SSE stream for chained video
 * generation. Same SSE framing as `/api/generate/stream` but with a
 * `ChainRequest` body and chain-shaped progress/complete events. */
export async function generateChainStream(
  req: ChainRequestWire,
  handlers: ChainStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  await postSseJsonStream<
    ChainRequestWire,
    ChainProgressEvent,
    SseChainCompleteEvent
  >({
    url: `${base}/api/generate/chain/stream`,
    body: req,
    signal,
    handlers,
    silentCloseMessage: "stream closed before completion",
  });
}

async function requireJson<T>(res: Response, label: string): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${label} failed: ${res.status} ${text}`.trim());
  }
  return (await res.json()) as T;
}

export async function createChainJob(
  req: ChainRequestWire,
): Promise<CreateChainJobResponse> {
  const res = await fetch(`${base}/api/chain-jobs`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  return requireJson<CreateChainJobResponse>(res, "POST /api/chain-jobs");
}

export async function listChainJobs(): Promise<ChainJobListing> {
  const res = await fetch(`${base}/api/chain-jobs`);
  return requireJson<ChainJobListing>(res, "GET /api/chain-jobs");
}

export async function getChainJob(id: string): Promise<ChainJobDetail> {
  const res = await fetch(`${base}/api/chain-jobs/${encodeURIComponent(id)}`);
  return requireJson<ChainJobDetail>(res, `GET /api/chain-jobs/${id}`);
}

export async function resumeChainJob(id: string): Promise<ChainJobSummary> {
  const res = await fetch(
    `${base}/api/chain-jobs/${encodeURIComponent(id)}/resume`,
    { method: "POST" },
  );
  return requireJson<ChainJobSummary>(res, `POST /api/chain-jobs/${id}/resume`);
}

export async function retakeChainJob(
  id: string,
  req: RetakeRequest,
): Promise<ChainJobSummary> {
  const res = await fetch(
    `${base}/api/chain-jobs/${encodeURIComponent(id)}/retake`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    },
  );
  return requireJson<ChainJobSummary>(res, `POST /api/chain-jobs/${id}/retake`);
}

export async function cancelChainJob(id: string): Promise<ChainJobSummary> {
  const res = await fetch(
    `${base}/api/chain-jobs/${encodeURIComponent(id)}/cancel`,
    { method: "POST" },
  );
  return requireJson<ChainJobSummary>(res, `POST /api/chain-jobs/${id}/cancel`);
}

export async function deleteChainJob(id: string): Promise<void> {
  const res = await fetch(`${base}/api/chain-jobs/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok && res.status !== 204) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `DELETE /api/chain-jobs/${id} failed: ${res.status} ${text}`.trim(),
    );
  }
}

export async function gcChainJobs(): Promise<{
  swept_ephemeral_jobs: number;
  pruned_artifact_dirs: number;
}> {
  const res = await fetch(`${base}/api/chain-jobs/gc`, { method: "POST" });
  return requireJson(res, "POST /api/chain-jobs/gc");
}

export function chainJobEventsUrl(id: string): string {
  return `${base}/api/chain-jobs/${encodeURIComponent(id)}/events`;
}

export function chainJobStagePreviewUrl(id: string, idx: number): string {
  return `${base}/api/chain-jobs/${encodeURIComponent(id)}/stages/${encodeURIComponent(
    String(idx),
  )}/preview`;
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
