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
  /** Called once the server has accepted the SSE response. */
  onOpen?: () => void;
  /** Called when the initial connection cannot be established. */
  onOpenError?: (error: Error) => void;
  /** Called when the stream ends or errors after retries. */
  onClose?: (error: Error | null) => void;
  /** Retry transient drops (default true for GET snapshots-first streams). */
  retry?: boolean;
  /** Additional request headers, merged with authentication and SSE defaults. */
  headers?: HeadersInit;
  /** Explicit host; defaults to the primary connection. */
  target?: ApiTarget;
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
      onopen(response) {
        if (!response.ok) {
          const error = new Error(`SSE request failed with HTTP ${response.status}`);
          options.onOpenError?.(error);
          throw error;
        }
        if (!opened) {
          opened = true;
          options.onOpen?.();
        }
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
