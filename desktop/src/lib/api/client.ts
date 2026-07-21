import { useConnectionStore } from "../../stores/connection";
import { ApiError, apiFetchTo, apiJsonTo, type ApiTarget } from "@studio/api/client";

export { ApiError, apiFetchTo, apiHeaders, apiJsonTo } from "@studio/api/client";
export type { ApiTarget } from "@studio/api/client";

/** Resolve the current engine target or throw a directed error. */
export function currentTarget(): ApiTarget {
  const conn = useConnectionStore();
  if (!conn.baseUrl) {
    throw new ApiError("No engine connected. Start the built-in engine in Settings.", 0);
  }
  return { baseUrl: conn.baseUrl, apiKey: conn.apiKey };
}

export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  return apiFetchTo(currentTarget(), path, init);
}

export async function apiJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  return apiJsonTo<T>(currentTarget(), path, init);
}
