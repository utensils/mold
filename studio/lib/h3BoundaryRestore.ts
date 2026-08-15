/**
 * FL2VA reuse-time boundary media restore. "Reuse settings" from an H3 print
 * restores its first/last frames as bytes-less reattach descriptors (gallery
 * metadata records a filename and digest, never payload bytes). When the
 * original was a gallery image on any connected host, the bytes are still
 * fetchable — this module finds the descriptors that need media and resolves
 * them by filename so the well fills itself instead of demanding a reattach.
 *
 * Fetching is caller-supplied (host enumeration and auth differ per surface);
 * committing is the caller's job too, so a slot the user reattached while a
 * fetch was in flight is never clobbered.
 */
import type { MinimaxH3AuthoringState } from "./minimaxH3Authoring";

export type H3BoundaryEndpointName = "firstFrame" | "lastFrame";

export interface H3BoundaryFetch {
  endpoint: H3BoundaryEndpointName;
  filename: string;
  base64: string;
}

export interface H3BoundaryFetchOutcome {
  restored: H3BoundaryFetch[];
  /** Filenames that were keyed for restore but not found anywhere. */
  failed: string[];
}

/** The slots of `authoring` that want media: present, bytes-less, named. */
export function h3BoundariesNeedingMedia(
  authoring: MinimaxH3AuthoringState | null | undefined,
): {
  endpoint: H3BoundaryEndpointName;
  filename: string;
  sha256: string | null;
}[] {
  const out: {
    endpoint: H3BoundaryEndpointName;
    filename: string;
    sha256: string | null;
  }[] = [];
  for (const endpoint of ["firstFrame", "lastFrame"] as const) {
    const slot = authoring?.[endpoint];
    if (slot && !slot.data && slot.filename.trim()) {
      out.push({
        endpoint,
        filename: slot.filename,
        sha256: slot.sha256 ?? null,
      });
    }
  }
  return out;
}

export async function fetchH3BoundaryMedia(
  authoring: MinimaxH3AuthoringState | null | undefined,
  fetchByFilename: (filename: string) => Promise<string | null>,
): Promise<H3BoundaryFetchOutcome> {
  const restored: H3BoundaryFetch[] = [];
  const failed: string[] = [];
  for (const want of h3BoundariesNeedingMedia(authoring)) {
    const base64 = await fetchByFilename(want.filename).catch(() => null);
    if (base64) {
      restored.push({
        endpoint: want.endpoint,
        filename: want.filename,
        base64,
      });
    } else {
      failed.push(want.filename);
    }
  }
  return { restored, failed };
}
