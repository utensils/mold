import { fetchEventSource } from "@microsoft/fetch-event-source";
import { apiHeaders, currentTarget, type ApiTarget } from "./client";

/**
 * SSE over fetch, everywhere — native EventSource can't send X-Api-Key and
 * several mold streams are POST (generate/upscale/chain).
 */
export interface StreamOptions {
  method?: "GET" | "POST";
  body?: unknown;
  signal: AbortSignal;
  onEvent: (event: string, data: string) => void;
  /** Called whenever the server accepts the SSE response, including reconnects. */
  onOpen?: (response: Response) => void;
  /** Called when the initial connection cannot be established. */
  onOpenError?: (error: Error) => void;
  /** Called when the stream ends or errors after retries. */
  onClose?: (error: Error | null) => void;
  /** Retry transient drops (default true for GET snapshots-first streams). */
  retry?: boolean;
  /** HTTP responses that are configuration errors, not reconnectable drops. */
  terminalHttpStatuses?: readonly number[];
  /** Additional request headers, merged with authentication and SSE defaults. */
  headers?: HeadersInit;
  /** Explicit host; defaults to the primary connection. */
  target?: ApiTarget;
}

async function responseError(response: Response): Promise<Error> {
  const fallback = `SSE request failed with HTTP ${response.status}`;
  let body: unknown;
  try {
    body = await response.clone().json();
  } catch {
    try {
      const text = (await response.clone().text()).trim();
      if (text) return Object.assign(new Error(text), { status: response.status });
    } catch {
      // Keep the status fallback when an intermediary supplies no readable body.
    }
  }
  if (typeof body === "object" && body !== null) {
    const record = body as Record<string, unknown>;
    const detail =
      (typeof record.error === "string" && record.error.trim()) ||
      (typeof record.message === "string" && record.message.trim());
    if (detail) {
      return Object.assign(new Error(detail), { status: response.status, body });
    }
  }
  return Object.assign(new Error(fallback), { status: response.status, body });
}

export async function sseStream(path: string, options: StreamOptions): Promise<void> {
  const target = options.target ?? currentTarget();
  const headers = apiHeaders(target, options.headers);
  headers.set("Accept", "text/event-stream");
  const method = options.method ?? "GET";
  let body: string | undefined;
  if (options.body !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.body);
  }
  const retriable = options.retry ?? method === "GET";
  let opened = false;

  try {
    await fetchEventSource(`${target.baseUrl}${path}`, {
      method,
      headers: Object.fromEntries(headers.entries()),
      ...(body !== undefined ? { body } : {}),
      signal: options.signal,
      openWhenHidden: true,
      async onopen(response) {
        if (!response.ok) {
          const error = await responseError(response);
          options.onOpenError?.(error);
          throw error;
        }
        opened = true;
        options.onOpen?.(response);
        return Promise.resolve();
      },
      onmessage(msg) {
        options.onEvent(msg.event || "message", msg.data);
      },
      onerror(err) {
        const error = err instanceof Error ? err : new Error(String(err));
        if (!opened && options.onOpenError) {
          options.onOpenError(error);
          throw error;
        }
        const status = (error as Error & { status?: number }).status;
        if (status !== undefined && options.terminalHttpStatuses?.includes(status)) throw error;
        if (!retriable) throw error;
        // returning undefined lets fetchEventSource retry with backoff
      },
    });
    options.onClose?.(null);
  } catch (err) {
    if (options.signal.aborted) {
      options.onClose?.(null);
      return;
    }
    options.onClose?.(err instanceof Error ? err : new Error(String(err)));
  }
}
