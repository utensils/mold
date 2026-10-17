/*
 * Explicit-target client for `GET /api/gallery/changes` (protocol 1) and the
 * fetch-backed `GalleryReplicaTransport` web, the phone and desktop's remote
 * buckets bind. The legacy arm is the existing conditional listing, so an
 * older host keeps answering 304s exactly as before.
 */
import {
  ApiError,
  apiFetchTo,
  conditionalApiJsonTo,
  type ApiTarget,
} from "./client";
import {
  GALLERY_CHANGES_PROTOCOL_VERSION,
  GalleryFeedUnavailableError,
  type GalleryChangesPage,
  type GalleryReplicaRow,
  type GalleryReplicaTransport,
} from "../lib/galleryReplica";

export const GALLERY_FEED_NOT_READY = "GALLERY_FEED_NOT_READY";
export const GALLERY_FEED_UNSUPPORTED = "GALLERY_FEED_UNSUPPORTED";

/** `capabilities.gallery.changes_feed` names protocol 1. Absent = an older
 * server (or a DB-less one): keep the full listing. */
export function galleryChangesFeedAdvertised(
  capabilities:
    | {
        gallery?: {
          changes_feed?: { protocol_version?: number | null } | null;
        } | null;
      }
    | null
    | undefined,
): boolean {
  return (
    capabilities?.gallery?.changes_feed?.protocol_version ===
    GALLERY_CHANGES_PROTOCOL_VERSION
  );
}

export function galleryChangesPath(
  since: string | null,
  limit: number,
): string {
  const query = new URLSearchParams();
  if (since !== null) query.set("since", since);
  query.set("limit", String(limit));
  return `/api/gallery/changes?${query.toString()}`;
}

function errorCode(body: unknown): string | null {
  return typeof body === "object" &&
    body !== null &&
    typeof (body as { code?: unknown }).code === "string"
    ? (body as { code: string }).code
    : null;
}

/** Validate a decoded page; anything else is not protocol 1. */
export function parseGalleryChangesPage<Row extends GalleryReplicaRow>(
  value: unknown,
): GalleryChangesPage<Row> {
  const page = value as Partial<GalleryChangesPage<Row>> | null;
  if (
    !page ||
    page.protocol_version !== GALLERY_CHANGES_PROTOCOL_VERSION ||
    typeof page.instance_id !== "string" ||
    typeof page.cursor !== "string" ||
    typeof page.more !== "boolean" ||
    typeof page.reset !== "boolean" ||
    (page.changes !== undefined && !Array.isArray(page.changes))
  )
    throw new GalleryFeedUnavailableError("unsupported");
  // `mold_core::GalleryChangesPage.changes` and `GalleryChange.removed` are
  // `#[serde(default)]`: absent reads as `[]` / `false`.
  return {
    ...(page as GalleryChangesPage<Row>),
    changes: (page.changes ?? []).map((change) => ({
      ...change,
      removed: change.removed === true,
    })),
  };
}

/** Map a feed refusal to the replica's fallback signal; rethrow the rest. */
export function galleryFeedFailure(error: unknown): never {
  if (error instanceof ApiError) {
    const code = errorCode(error.body);
    if (error.status === 503 && code === GALLERY_FEED_NOT_READY)
      throw new GalleryFeedUnavailableError("not_ready");
    if (
      code === GALLERY_FEED_UNSUPPORTED ||
      error.status === 404 ||
      error.status === 405
    )
      throw new GalleryFeedUnavailableError("unsupported");
  }
  throw error;
}

export async function fetchGalleryChanges<Row extends GalleryReplicaRow>(
  target: ApiTarget,
  since: string | null,
  limit: number,
  init: RequestInit = {},
): Promise<GalleryChangesPage<Row>> {
  let response: Response;
  try {
    response = await apiFetchTo(target, galleryChangesPath(since, limit), init);
  } catch (error) {
    galleryFeedFailure(error);
  }
  return parseGalleryChangesPage<Row>(await response.json());
}

export interface HttpGalleryReplicaOptions {
  /** Whether the host advertises the feed (usually its cached capabilities). */
  feedSupported: () => boolean | Promise<boolean>;
  /** Whether the legacy path should list `?view=trash` too. Omit to list
   * only the library (trash rows already held are then preserved). */
  listTrash?: () => boolean | Promise<boolean>;
  /** Per-request deadline in milliseconds. */
  timeoutMs?: number;
}

/** A replica transport over plain HTTP. `target` is resolved per request so
 * a re-keyed host never reads with stale credentials. */
export function httpGalleryReplicaTransport<Row extends GalleryReplicaRow>(
  target: ApiTarget | (() => ApiTarget),
  options: HttpGalleryReplicaOptions,
): GalleryReplicaTransport<Row> {
  const resolveTarget = typeof target === "function" ? target : () => target;
  const init = (): RequestInit =>
    options.timeoutMs !== undefined &&
    typeof AbortSignal !== "undefined" &&
    typeof AbortSignal.timeout === "function"
      ? { signal: AbortSignal.timeout(options.timeoutMs) }
      : {};
  const transport: GalleryReplicaTransport<Row> = {
    feedSupported: options.feedSupported,
    changes: (since, limit) =>
      fetchGalleryChanges<Row>(resolveTarget(), since, limit, init()),
    legacyLibrary: () =>
      conditionalApiJsonTo<Row[]>(resolveTarget(), "/api/gallery", init()),
  };
  const listTrash = options.listTrash;
  if (listTrash)
    transport.legacyTrash = async () =>
      (await listTrash())
        ? conditionalApiJsonTo<Row[]>(
            resolveTarget(),
            "/api/gallery?view=trash",
            init(),
          )
        : null;
  return transport;
}
