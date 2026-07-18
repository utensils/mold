import { apiFetch, apiFetchTo, apiJson, apiJsonTo, type ApiTarget } from "./client";
import type { ModelComponentsResponse } from "./types";

/** Load a model onto the GPU. */
export function loadModel(model: string, target?: ApiTarget): Promise<Response> {
  const init: RequestInit = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  };
  return target ? apiFetchTo(target, "/api/models/load", init) : apiFetch("/api/models/load", init);
}

/** Unload a model from the GPU. */
export function unloadModel(model: string, target?: ApiTarget): Promise<Response> {
  const init: RequestInit = {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  };
  return target
    ? apiFetchTo(target, "/api/models/unload", init)
    : apiFetch("/api/models/unload", init);
}

/** Per-component presence for an installed model. */
export function fetchModelComponents(
  model: string,
  target?: ApiTarget,
): Promise<ModelComponentsResponse> {
  const path = `/api/models/${encodeURIComponent(model)}/components`;
  return target
    ? apiJsonTo<ModelComponentsResponse>(target, path)
    : apiJson<ModelComponentsResponse>(path);
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
export function removeModel(model: string, target?: ApiTarget): Promise<ModelRemovalResponse> {
  const path = `/api/models/${encodeURIComponent(model)}`;
  const init: RequestInit = { method: "DELETE" };
  return target
    ? apiJsonTo<ModelRemovalResponse>(target, path, init)
    : apiJson<ModelRemovalResponse>(path, init);
}
