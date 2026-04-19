import type {
  ExpandRequestWire,
  ExpandResponseWire,
  GalleryImage,
  GenerateRequestWire,
  ModelInfoExtended,
  ServerCapabilities,
  ServerStatus,
  SseCompleteEvent,
  SseProgressEvent,
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
  if (res.status === 403) {
    throw new Error(
      "Delete is disabled on this server (set MOLD_GALLERY_ALLOW_DELETE=1 on the host).",
    );
  }
  if (!res.ok && res.status !== 204) {
    throw new Error(`DELETE failed: ${res.status} ${res.statusText}`);
  }
}

/**
 * Fetch server capabilities. The SPA uses these to decide which UI
 * affordances to surface — e.g. hiding the delete button when the operator
 * hasn't opted in. Returns safe defaults if the server is too old to know
 * about the endpoint.
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
  return { gallery: { can_delete: false } };
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
          handlers.onComplete(parsed as SseCompleteEvent);
        } else if (evt.event === "error") {
          const body = evt.data;
          handlers.onError({ kind: "http", status: 0, body });
        } else {
          // Default `progress` event — the server emits many progress types
          // tagged with an internal `type` discriminator.
          handlers.onProgress(parsed as SseProgressEvent);
        }
      },
      onHttpError: (res) => {
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
  } catch (err) {
    if (signal?.aborted) return; // user canceled
    handlers.onError({
      kind: "network",
      message: err instanceof Error ? err.message : String(err),
    });
  }
}
