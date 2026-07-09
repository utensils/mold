import { apiFetch, apiJson } from "./client";
import { isCatalogId } from "../catalog";
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

export function searchCatalog(params: CatalogSearchParams): Promise<CatalogSearchResponse> {
  const query = new URLSearchParams();
  if (params.q) query.set("q", params.q);
  if (params.family) query.set("family", params.family);
  if (params.kind) query.set("kind", params.kind);
  if (params.source) query.set("source", params.source);
  if (params.include_nsfw != null) query.set("include_nsfw", String(params.include_nsfw));
  if (params.page != null) query.set("page", String(params.page));
  if (params.page_size != null) query.set("page_size", String(params.page_size));
  return apiJson<CatalogSearchResponse>(`/api/catalog/search?${query.toString()}`);
}

export async function fetchCatalogFamilies(): Promise<string[]> {
  const res = await apiJson<{ families: CatalogFamily[] }>("/api/catalog/families");
  return res.families.map((f) => f.family);
}

/**
 * Start a pull. Catalog ids (`cv:` / `hf:`) route through the catalog download
 * dispatcher, which also enqueues shared companions; plain model names go
 * straight to the download queue. The id is placed raw in the path (its colons
 * and slashes are part of the wildcard match — do not URL-encode it).
 */
export async function startCatalogDownload(id: string): Promise<void> {
  if (isCatalogId(id)) {
    await apiFetch(`/api/catalog/${id}/download`, { method: "POST" });
    return;
  }
  await apiFetch("/api/downloads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: id }),
  });
}

export type { CatalogDownloadResponse };
