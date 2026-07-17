import { apiFetchTo, apiJsonTo, currentTarget, type ApiTarget } from "./client";
import { isCatalogId } from "../catalog";
import { catalogCredentialHeaders } from "../catalogCredentials";
import type { CatalogDownloadResponse, CatalogFamily, CatalogSearchResponse } from "./types";

export interface CatalogSearchParams {
  q?: string | undefined;
  family?: string | undefined;
  kind?: string | undefined;
  source?: string | undefined;
  include_nsfw?: boolean | undefined;
  page?: number | undefined;
  page_size?: number | undefined;
}

export async function searchCatalog(
  params: CatalogSearchParams,
  forwardCredentials = false,
): Promise<CatalogSearchResponse> {
  const query = new URLSearchParams();
  if (params.q) query.set("q", params.q);
  if (params.family) query.set("family", params.family);
  if (params.kind) query.set("kind", params.kind);
  if (params.source) query.set("source", params.source);
  if (params.include_nsfw != null) query.set("include_nsfw", String(params.include_nsfw));
  if (params.page != null) query.set("page", String(params.page));
  if (params.page_size != null) query.set("page_size", String(params.page_size));
  const headers = await catalogCredentialHeaders(forwardCredentials);
  return apiJsonTo<CatalogSearchResponse>(
    currentTarget(),
    `/api/catalog/search?${query.toString()}`,
    { headers },
  );
}

export async function fetchCatalogFamilies(forwardCredentials = false): Promise<string[]> {
  const headers = await catalogCredentialHeaders(forwardCredentials);
  const res = await apiJsonTo<{ families: CatalogFamily[] }>(
    currentTarget(),
    "/api/catalog/families",
    {
      headers,
    },
  );
  return res.families.map((f) => f.family);
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
): Promise<void> {
  const headers = await catalogCredentialHeaders(forwardCredentials);
  if (isCatalogId(id)) {
    await apiFetchTo(target, `/api/catalog/${id}/download`, { method: "POST", headers });
    return;
  }
  headers.set("Content-Type", "application/json");
  await apiFetchTo(target, "/api/downloads", {
    method: "POST",
    headers,
    body: JSON.stringify({ model: id }),
  });
}

export type { CatalogDownloadResponse };
