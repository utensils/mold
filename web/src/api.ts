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
  GenerateRequestWire,
  GenerationMemoryEstimate,
  ModelInfoExtended,
  ModelComponentsResponse,
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
import type {
  AmendRequest as AmendRequestWire,
  AmendResponse as AmendResponseWire,
  ChainValidationResponse,
} from "@studio/lib/api/chainTypes";
export type {
  ChainValidationResponse,
  ChainValidationStage,
} from "@studio/lib/api/chainTypes";

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
  supports_sequence: boolean;
  sequence_unsupported_reason?: string | null;
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
export async function fetchQueue(
  target?: StreamTarget,
  signal?: AbortSignal,
): Promise<QueueListing> {
  const headers = targetHeaders(target);
  const res = await fetch(`${targetBase(target)}/api/queue`, {
    ...(Object.keys(headers).length ? { headers } : {}),
    signal,
  });
  if (!res.ok) throw new Error(`GET /api/queue failed: ${res.status}`);
  return (await res.json()) as QueueListing;
}

/** Cancel a queued or running generation on the exact host that owns it. */
export async function cancelQueueJob(
  id: string,
  target?: StreamTarget,
): Promise<void> {
  const res = await fetch(
    `${targetBase(target)}/api/queue/${encodeURIComponent(id)}`,
    { method: "DELETE", headers: targetHeaders(target) },
  );
  if (!res.ok && res.status !== 404) {
    const detail = await res.text().catch(() => "");
    throw new Error(
      `Cancel failed (${res.status})${detail ? `: ${detail}` : ""}`,
    );
  }
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

/**
 * Move a queued job to a new zero-based position via `PATCH /api/queue/:id`
 * with a `position` field. Callers check `capabilities.queue.can_reorder`
 * because queue reordering is an optional current-server capability.
 */
export async function reorderQueueJob(
  id: string,
  position: number,
  signal?: AbortSignal,
): Promise<QueueEntry> {
  const res = await fetch(`${base}/api/queue/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ position }),
    signal,
  });
  if (!res.ok) throw new Error(`PATCH /api/queue/${id} failed: ${res.status}`);
  return (await res.json()) as QueueEntry;
}

const chainLimitsCache = new Map<string, { value: ChainLimits; at: number }>();
const CHAIN_LIMITS_TTL_MS = 30_000;

export async function fetchChainLimits(
  model: string,
  target?: StreamTarget,
): Promise<ChainLimits> {
  const now = Date.now();
  // Limits are per model AND per host — a remote's checkpoint may chain
  // where the origin's doesn't.
  const key = `${targetBase(target)}|${model}`;
  const cached = chainLimitsCache.get(key);
  if (cached && now - cached.at < CHAIN_LIMITS_TTL_MS) return cached.value;
  const res = await fetch(
    `${targetBase(target)}/api/capabilities/chain-limits?model=${encodeURIComponent(model)}`,
    { headers: targetHeaders(target) },
  );
  if (!res.ok) throw new Error(`chain-limits fetch failed: ${res.status}`);
  const value: ChainLimits = await res.json();
  chainLimitsCache.set(key, { value, at: now });
  return value;
}

export async function expandPrompt(
  req: ExpandRequestWire,
  signal?: AbortSignal,
  target?: StreamTarget,
): Promise<ExpandResponseWire> {
  const res = await fetch(`${targetBase(target)}/api/expand`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...targetHeaders(target) },
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

/**
 * Where a generation stream is dispatched. Omitted = the serving origin (the
 * single-host case and every non-routed caller). The key travels as the
 * server's `x-api-key` header — never in the URL.
 */
export interface StreamTarget {
  baseUrl: string;
  apiKey?: string;
}

function targetBase(target?: StreamTarget): string {
  return target?.baseUrl ?? base;
}

function targetHeaders(target?: StreamTarget): Record<string, string> {
  return target?.apiKey ? { "x-api-key": target.apiKey } : {};
}

/** Normalize and validate a sequence on its exact rendering host without
 * creating a durable job or starting any downloads. */
export async function validateChain(
  req: ChainRequestWire,
  target?: StreamTarget,
): Promise<ChainValidationResponse> {
  const res = await fetch(`${targetBase(target)}/api/generate/chain/validate`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...targetHeaders(target),
    },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const body = await res
      .clone()
      .json()
      .catch(() => null);
    const detail =
      typeof body === "object" && body !== null && "error" in body
        ? String((body as { error: unknown }).error)
        : await res.text().catch(() => "");
    throw new ApiHttpError("Sequence validation", res.status, detail);
  }
  return (await res.json()) as ChainValidationResponse;
}

/** Advisory VRAM preflight against the machine that will render the print. */
export async function fetchGenerationEstimate(
  req: GenerateRequestWire,
  target?: StreamTarget,
): Promise<GenerationMemoryEstimate> {
  const res = await fetch(`${targetBase(target)}/api/generate/estimate`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...targetHeaders(target),
    },
    body: JSON.stringify(req),
  });
  return requireJson<GenerationMemoryEstimate>(
    res,
    "POST /api/generate/estimate",
  );
}

export async function generateStream(
  req: GenerateRequestWire,
  handlers: GenerateStreamHandlers,
  signal?: AbortSignal,
  target?: StreamTarget,
): Promise<void> {
  await postSseJsonStream<
    GenerateRequestWire,
    SseProgressEvent,
    SseCompleteEvent
  >({
    url: `${targetBase(target)}/api/generate/stream`,
    body: req,
    signal,
    headers: targetHeaders(target),
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
  target?: StreamTarget,
): Promise<void> {
  await postSseJsonStream<
    UpscaleRequestWire,
    SseProgressEvent,
    SseUpscaleCompleteEvent
  >({
    url: `${targetBase(target)}/api/upscale/stream`,
    body: req,
    signal,
    headers: targetHeaders(target),
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
  target?: StreamTarget,
): Promise<void> {
  await postSseJsonStream<
    ChainRequestWire,
    ChainProgressEvent,
    SseChainCompleteEvent
  >({
    url: `${targetBase(target)}/api/generate/chain/stream`,
    body: req,
    signal,
    headers: targetHeaders(target),
    handlers,
    silentCloseMessage: "stream closed before completion",
  });
}

/** HTTP failure that keeps the status reachable — amend conflict handling
 * (409 → create-as-new fallback) branches on it. */
export class ApiHttpError extends Error {
  constructor(
    label: string,
    public readonly status: number,
    body: string,
  ) {
    super(`${label} failed: ${status} ${body}`.trim());
    this.name = "ApiHttpError";
  }
}

async function requireJson<T>(res: Response, label: string): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new ApiHttpError(label, res.status, text);
  }
  return (await res.json()) as T;
}

export async function createChainJob(
  req: ChainRequestWire,
  target?: StreamTarget,
): Promise<CreateChainJobResponse> {
  const res = await fetch(`${targetBase(target)}/api/chain-jobs`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...targetHeaders(target),
    },
    body: JSON.stringify(req),
  });
  return requireJson<CreateChainJobResponse>(res, "POST /api/chain-jobs");
}

export async function listChainJobs(
  target?: StreamTarget,
): Promise<ChainJobListing> {
  const res = await fetch(`${targetBase(target)}/api/chain-jobs`, {
    headers: targetHeaders(target),
  });
  return requireJson<ChainJobListing>(res, "GET /api/chain-jobs");
}

export async function getChainJob(
  id: string,
  target?: StreamTarget,
): Promise<ChainJobDetail> {
  const res = await fetch(
    `${targetBase(target)}/api/chain-jobs/${encodeURIComponent(id)}`,
    { headers: targetHeaders(target) },
  );
  return requireJson<ChainJobDetail>(res, `GET /api/chain-jobs/${id}`);
}

export async function resumeChainJob(
  id: string,
  target?: StreamTarget,
): Promise<ChainJobSummary> {
  const res = await fetch(
    `${targetBase(target)}/api/chain-jobs/${encodeURIComponent(id)}/resume`,
    { method: "POST", headers: targetHeaders(target) },
  );
  return requireJson<ChainJobSummary>(res, `POST /api/chain-jobs/${id}/resume`);
}

export async function retakeChainJob(
  id: string,
  req: RetakeRequest,
  target?: StreamTarget,
): Promise<ChainJobSummary> {
  const res = await fetch(
    `${targetBase(target)}/api/chain-jobs/${encodeURIComponent(id)}/retake`,
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...targetHeaders(target),
      },
      body: JSON.stringify(req),
    },
  );
  return requireJson<ChainJobSummary>(res, `POST /api/chain-jobs/${id}/retake`);
}

export async function cancelChainJob(
  id: string,
  target?: StreamTarget,
): Promise<ChainJobSummary> {
  const res = await fetch(
    `${targetBase(target)}/api/chain-jobs/${encodeURIComponent(id)}/cancel`,
    { method: "POST", headers: targetHeaders(target) },
  );
  return requireJson<ChainJobSummary>(res, `POST /api/chain-jobs/${id}/cancel`);
}

export async function deleteChainJob(
  id: string,
  target?: StreamTarget,
): Promise<void> {
  const res = await fetch(
    `${targetBase(target)}/api/chain-jobs/${encodeURIComponent(id)}`,
    { method: "DELETE", headers: targetHeaders(target) },
  );
  if (!res.ok && res.status !== 204) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `DELETE /api/chain-jobs/${id} failed: ${res.status} ${text}`.trim(),
    );
  }
}

/** POST /api/chain-jobs/:id/amend — full edited stage list; the server
 * recomputes the preserved prefix and re-renders only dirty stages. A 409
 * (`ApiHttpError.status`) means the job has moved on and the edit must be
 * created as a new sequence instead. */
export async function amendChainJob(
  id: string,
  req: AmendRequestWire,
  target?: StreamTarget,
): Promise<AmendResponseWire> {
  const res = await fetch(
    `${targetBase(target)}/api/chain-jobs/${encodeURIComponent(id)}/amend`,
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...targetHeaders(target),
      },
      body: JSON.stringify(req),
    },
  );
  return requireJson<AmendResponseWire>(
    res,
    `POST /api/chain-jobs/${id}/amend`,
  );
}

export async function gcChainJobs(target?: StreamTarget): Promise<{
  swept_ephemeral_jobs: number;
  pruned_artifact_dirs: number;
}> {
  const res = await fetch(`${targetBase(target)}/api/chain-jobs/gc`, {
    method: "POST",
    headers: targetHeaders(target),
  });
  return requireJson(res, "POST /api/chain-jobs/gc");
}

export function chainJobEventsUrl(id: string, target?: StreamTarget): string {
  return `${targetBase(target)}/api/chain-jobs/${encodeURIComponent(id)}/events`;
}

export function chainJobStagePreviewUrl(
  id: string,
  idx: number,
  target?: StreamTarget,
): string {
  return `${targetBase(target)}/api/chain-jobs/${encodeURIComponent(id)}/stages/${encodeURIComponent(
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
  if (!r.ok) {
    const detail = await r.text().catch(() => "");
    throw new Error(detail || `/api/catalog/${id}/download ${r.status}`);
  }
  return r.json();
}

export type CatalogCredentialProvider = "hf" | "civitai";
export type CatalogCredentialSource = "environment" | "server" | null;

export interface CatalogCredentialState {
  configured: boolean;
  source: CatalogCredentialSource;
  masked: string | null;
}

export interface CatalogCredentialStatus {
  hf: CatalogCredentialState;
  civitai: CatalogCredentialState;
}

async function catalogCredentialResponse(
  response: Response,
  action: string,
): Promise<CatalogCredentialStatus> {
  if (response.ok) return response.json();
  const body = await response.text().catch(() => "");
  if (body) {
    try {
      const parsed = JSON.parse(body) as { error?: string };
      throw new Error(parsed.error || body);
    } catch (error) {
      if (error instanceof SyntaxError) throw new Error(body);
      throw error;
    }
  }
  throw new Error(`${action} failed: ${response.status}`);
}

export async function getCatalogCredentialStatus(): Promise<CatalogCredentialStatus> {
  return catalogCredentialResponse(
    await fetch("/api/catalog/credentials"),
    "Loading catalog credentials",
  );
}

export async function putCatalogCredential(
  provider: CatalogCredentialProvider,
  token: string,
): Promise<CatalogCredentialStatus> {
  return catalogCredentialResponse(
    await fetch(`/api/catalog/credentials/${provider}`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ token }),
    }),
    "Saving catalog credential",
  );
}

export async function deleteCatalogCredential(
  provider: CatalogCredentialProvider,
): Promise<CatalogCredentialStatus> {
  return catalogCredentialResponse(
    await fetch(`/api/catalog/credentials/${provider}`, {
      method: "DELETE",
    }),
    "Removing catalog credential",
  );
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

// ─── Installed-model management ─────────────────────────────────────────────
// Thin wrappers over the server's model routes, consumed by the Models
// workspace detail drawer (Load / Unload / Delete).

/// Response of `DELETE /api/models/:model` — what was deleted, what was
/// kept for other models, and how many bytes were actually freed on disk.
/// Mirrors `mold_core::ModelRemovalResponse`.
export interface ModelRemovalResult {
  removed: string[];
  kept: { name: string; kept_for: string[] }[];
  freed_bytes: number;
}

/// Load a downloaded model into GPU memory (`POST /api/models/load`).
export async function loadModel(model: string, gpu?: number): Promise<void> {
  const body: { model: string; gpu?: number } = { model };
  if (gpu !== undefined) body.gpu = gpu;
  const res = await fetch(`${base}/api/models/load`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST /api/models/load failed: ${res.status}`);
}

/// Unload a model (or the active model when `model` is omitted) from GPU
/// memory (`DELETE /api/models/unload`).
export async function unloadModel(model?: string): Promise<void> {
  const res = await fetch(`${base}/api/models/unload`, {
    method: "DELETE",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(model ? { model } : {}),
  });
  if (!res.ok)
    throw new Error(`DELETE /api/models/unload failed: ${res.status}`);
}

/// Delete a downloaded model's on-disk files (`DELETE /api/models/:model`).
/// Rejects with a 409-derived error when the model is currently loaded.
export async function deleteModel(model: string): Promise<ModelRemovalResult> {
  const res = await fetch(`${base}/api/models/${encodeURIComponent(model)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const body = (await res.json().catch(() => null)) as {
      code?: string;
      error?: string;
    } | null;
    if (body?.code === "MODEL_LOADED") {
      throw new Error(`Unload ${model} before deleting it.`);
    }
    throw new Error(
      body?.error ?? `DELETE /api/models/${model} failed: ${res.status}`,
    );
  }
  return (await res.json()) as ModelRemovalResult;
}

/// Prompt history for ↑/↓ recall in the Create composer (`GET /api/history`,
/// newest-first). Returns just the prompt strings; the server also carries the
/// model + timestamp, which the composer doesn't need. Never throws — a server
/// without the history route just yields an empty list so ↑/↓ falls back to
/// plain caret movement.
export async function fetchPromptHistory(limit = 100): Promise<string[]> {
  try {
    const res = await fetch(`${base}/api/history?limit=${limit}`);
    if (!res.ok) return [];
    const body = (await res.json()) as {
      entries?: { prompt?: string }[];
    };
    return (body.entries ?? [])
      .map((e) => (e.prompt ?? "").trim())
      .filter((p) => p.length > 0);
  } catch {
    return [];
  }
}
