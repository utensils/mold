import { fetchEventSource } from "@microsoft/fetch-event-source";
import { apiHeaders, currentTarget } from "./client";

/**
 * SSE over fetch, everywhere — native EventSource can't send X-Api-Key and
 * several mold streams are POST (generate/upscale/chain).
 */
export interface StreamOptions {
  method?: "GET" | "POST";
  body?: unknown;
  signal: AbortSignal;
  onEvent: (event: string, data: string) => void;
  /** Called when the stream ends or errors after retries. */
  onClose?: (error: Error | null) => void;
  /** Retry transient drops (default true for GET snapshots-first streams). */
  retry?: boolean;
}

export async function sseStream(path: string, options: StreamOptions): Promise<void> {
  const target = currentTarget();
  const headers = apiHeaders(target, { Accept: "text/event-stream" });
  const method = options.method ?? "GET";
  let body: string | undefined;
  if (options.body !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.body);
  }
  const retriable = options.retry ?? method === "GET";

  try {
    await fetchEventSource(`${target.baseUrl}${path}`, {
      method,
      headers: Object.fromEntries(headers.entries()),
      ...(body !== undefined ? { body } : {}),
      signal: options.signal,
      openWhenHidden: true,
      onmessage(msg) {
        options.onEvent(msg.event || "message", msg.data);
      },
      onerror(err) {
        if (!retriable) throw err instanceof Error ? err : new Error(String(err));
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
