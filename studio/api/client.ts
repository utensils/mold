export interface ApiTarget {
  baseUrl: string;
  apiKey: string | null;
}

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

export class IncompatibleHostError extends ApiError {
  constructor(public readonly missingFields: readonly string[]) {
    super(
      `This Mold host is out of date. Upgrade it before reconnecting (missing: ${missingFields.join(
        ", ",
      )}).`,
      426,
    );
    this.name = "IncompatibleHostError";
  }
}

export function apiHeaders(target: ApiTarget, extra?: HeadersInit): Headers {
  const headers = new Headers(extra);
  if (target.apiKey) headers.set("X-Api-Key", target.apiKey);
  return headers;
}

export async function apiFetchTo(
  target: ApiTarget,
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const response = await fetch(`${target.baseUrl}${path}`, {
    ...init,
    headers: apiHeaders(target, init.headers),
  });
  if (!response.ok) {
    let body: unknown = null;
    try {
      body = await response.clone().json();
    } catch {
      // Non-JSON failures still retain their HTTP status and status text.
    }
    const detail =
      typeof body === "object" && body !== null && "error" in body
        ? String((body as { error: unknown }).error)
        : response.statusText;
    throw new ApiError(detail, response.status, body);
  }
  return response;
}

export async function apiJsonTo<T>(
  target: ApiTarget,
  path: string,
  init: RequestInit = {},
): Promise<T> {
  return (await apiFetchTo(target, path, init)).json() as Promise<T>;
}

export interface CurrentServerStatus {
  instance_id: string;
  hostname: string;
  queue_depth: number;
  gpu_info: { backend: string | null } | null;
}

export function parseCurrentServerStatus(value: unknown): CurrentServerStatus {
  const row = typeof value === "object" && value !== null ? value : {};
  const missing: string[] = [];
  const has = (key: string, predicate: (candidate: unknown) => boolean) => {
    const candidate = (row as Record<string, unknown>)[key];
    if (!predicate(candidate)) missing.push(key);
  };
  has("instance_id", (candidate) => typeof candidate === "string" && candidate.length > 0);
  has("hostname", (candidate) => typeof candidate === "string" && candidate.length > 0);
  has("queue_depth", (candidate) => typeof candidate === "number");
  if (missing.length > 0) throw new IncompatibleHostError(missing);
  return row as CurrentServerStatus;
}
