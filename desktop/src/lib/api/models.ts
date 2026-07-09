import { apiFetch, apiJson } from "./client";
import type { ModelComponentsResponse } from "./types";

/** Load a model onto the GPU. */
export function loadModel(model: string): Promise<Response> {
  return apiFetch("/api/models/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
}

/** Unload a model from the GPU. */
export function unloadModel(model: string): Promise<Response> {
  return apiFetch("/api/models/unload", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
}

/** Per-component presence for an installed model. */
export function fetchModelComponents(model: string): Promise<ModelComponentsResponse> {
  return apiJson<ModelComponentsResponse>(`/api/models/${encodeURIComponent(model)}/components`);
}

export interface KeptComponent {
  component: string;
  used_by: string[];
}

export interface ModelRemovalResponse {
  removed: string[];
  kept: KeptComponent[];
  freed_bytes: number;
}

/** Remove a model from disk. Shared components stay while referenced. */
export function removeModel(model: string): Promise<ModelRemovalResponse> {
  return apiJson<ModelRemovalResponse>(`/api/models/${encodeURIComponent(model)}`, {
    method: "DELETE",
  });
}
