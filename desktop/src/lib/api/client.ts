import { useConnectionStore } from "../../stores/connection";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly body: unknown = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface ApiTarget {
  baseUrl: string;
  apiKey: string | null;
}

/** Resolve the current engine target or throw a directed error. */
export function currentTarget(): ApiTarget {
  const conn = useConnectionStore();
  if (!conn.baseUrl) {
    throw new ApiError("No engine connected. Start the built-in engine in Settings.", 0);
  }
  return { baseUrl: conn.baseUrl, apiKey: conn.apiKey };
}

export function apiHeaders(target: ApiTarget, extra?: HeadersInit): Headers {
  const headers = new Headers(extra);
  if (target.apiKey) headers.set("X-Api-Key", target.apiKey);
  return headers;
}

export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const target = currentTarget();
  const response = await fetch(`${target.baseUrl}${path}`, {
    ...init,
    headers: apiHeaders(target, init.headers),
  });
  if (!response.ok) {
    let body: unknown = null;
    try {
      body = await response.clone().json();
    } catch {
      /* non-JSON error body */
    }
    const detail =
      typeof body === "object" && body !== null && "error" in body
        ? String((body as { error: unknown }).error)
        : response.statusText;
    throw new ApiError(detail, response.status, body);
  }
  return response;
}

export async function apiJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await apiFetch(path, init);
  return (await response.json()) as T;
}
