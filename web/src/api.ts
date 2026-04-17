import type { GalleryImage, ServerCapabilities } from "./types";

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
