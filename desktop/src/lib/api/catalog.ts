import { apiFetchTo, apiJsonTo, currentTarget, type ApiTarget } from "./client";
import { isCatalogId } from "../catalog";
import { catalogCredentialHeaders } from "../catalogCredentials";
import type {
  CatalogDownloadResponse,
  CatalogEntry,
  CatalogFamily,
  CatalogSearchResponse,
} from "./types";

export interface CatalogSearchParams {
  q?: string | undefined;
  family?: string | undefined;
  kind?: string | undefined;
  source?: string | undefined;
  include_nsfw?: boolean | undefined;
  sort?: string | undefined;
  page?: number | undefined;
  page_size?: number | undefined;
}

export async function searchCatalog(
  params: CatalogSearchParams,
  forwardCredentials = false,
  target?: ApiTarget,
): Promise<CatalogSearchResponse> {
  const query = new URLSearchParams();
  if (params.q) query.set("q", params.q);
  if (params.family) query.set("family", params.family);
  if (params.kind) query.set("kind", params.kind);
  if (params.source) query.set("source", params.source);
  if (params.include_nsfw != null) query.set("include_nsfw", String(params.include_nsfw));
  if (params.sort) query.set("sort", params.sort);
  if (params.page != null) query.set("page", String(params.page));
  if (params.page_size != null) query.set("page_size", String(params.page_size));
  const headers = await catalogCredentialHeaders(forwardCredentials);
  return apiJsonTo<CatalogSearchResponse>(
    target ?? currentTarget(),
    `/api/catalog/search?${query.toString()}`,
    { headers },
  );
}

export async function fetchCatalogFamilies(
  forwardCredentials = false,
  target?: ApiTarget,
): Promise<string[]> {
  const headers = await catalogCredentialHeaders(forwardCredentials);
  const res = await apiJsonTo<{ families: CatalogFamily[] }>(
    target ?? currentTarget(),
    "/api/catalog/families",
    {
      headers,
    },
  );
  return res.families.map((f) => f.family);
}

export interface CatalogInstalledParams {
  family?: string | undefined;
  kind?: string | undefined;
}

/**
 * Enumerate installed catalog entries — `GET /api/catalog/installed`, backed
 * by the per-install sidecar files under the host's `models_dir` (so no
 * upstream credentials are needed). `family` / `kind` are server-side filters
 * (e.g. `kind: "control-net"` for the ControlNet picker); entries carry
 * `installed` + `primary_path`. Same envelope shape as `/api/catalog/search`.
 */
export async function fetchCatalogInstalled(
  params: CatalogInstalledParams,
  target?: ApiTarget,
): Promise<CatalogSearchResponse> {
  const query = new URLSearchParams();
  if (params.family) query.set("family", params.family);
  if (params.kind) query.set("kind", params.kind);
  const qs = query.toString();
  return apiJsonTo<CatalogSearchResponse>(
    target ?? currentTarget(),
    `/api/catalog/installed${qs ? `?${qs}` : ""}`,
  );
}

/**
 * Fetch one catalog entry's full detail — `GET /api/catalog/:id`. The single
 * code path for the endpoint: the detail drawer reads everything, lazy card
 * sizes (`lib/catalogSizes.ts`) read through it keeping only `size_bytes`.
 * The id goes RAW in the path (colons and slashes are part of the wildcard
 * route match — do not URL-encode it).
 */
export async function fetchCatalogDetail(
  id: string,
  forwardCredentials = false,
  target?: ApiTarget,
): Promise<CatalogEntry> {
  const headers = await catalogCredentialHeaders(forwardCredentials);
  return apiJsonTo<CatalogEntry>(target ?? currentTarget(), `/api/catalog/${id}`, { headers });
}

/**
 * Start a pull. Catalog ids (`cv:` / `hf:`) route through the catalog download
 * dispatcher, which also enqueues shared companions; plain model names go
 * straight to the download queue. The id is placed raw in the path (its colons
 * and slashes are part of the wildcard match — do not URL-encode it).
 */
export async function startCatalogDownload(
  id: string,
  target: ApiTarget = currentTarget(),
  forwardCredentials = false,
): Promise<string | null> {
  const headers = await catalogCredentialHeaders(forwardCredentials);
  if (isCatalogId(id)) {
    const res = await apiFetchTo(target, `/api/catalog/${id}/download`, {
      method: "POST",
      headers,
    });
    // The primary job id lets pull-and-resume watch THIS pull exactly.
    const body = (await res.json().catch(() => null)) as CatalogDownloadResponse | null;
    const jobId = body?.primary_job_id ?? body?.companion_jobs?.[0]?.job_id ?? null;
    if (!jobId) {
      throw new Error("The host accepted the pull but did not queue any files.");
    }
    return jobId;
  }
  headers.set("Content-Type", "application/json");
  const res = await apiFetchTo(target, "/api/downloads", {
    method: "POST",
    headers,
    body: JSON.stringify({ model: id }),
  });
  const body = (await res.json().catch(() => null)) as { id?: string } | null;
  return body?.id ?? null;
}

export type { CatalogDownloadResponse };
