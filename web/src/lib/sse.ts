/**
 * POST-capable Server-Sent Events client.
 *
 * Rationale: the browser's built-in EventSource is GET-only, but
 * /api/generate/stream needs a JSON request body. We stream the response,
 * buffer by newline, and reassemble SSE events (lines of `event:` and
 * `data:` terminated by a blank line).
 */

export interface SseEvent {
  event: string | null;
  data: string;
}

export interface StreamSseOptions<TBody> {
  url: string;
  body: TBody;
  signal?: AbortSignal;
  onEvent: (evt: SseEvent) => void;
  /** Called once with the Response if the server returned a non-2xx. */
  onHttpError?: (res: Response) => void;
}

export async function streamSse<TBody>(
  opts: StreamSseOptions<TBody>,
): Promise<Response> {
  const res = await fetch(opts.url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(opts.body),
    signal: opts.signal,
  });

  if (!res.ok) {
    opts.onHttpError?.(res);
    return res;
  }
  if (!res.body) {
    throw new Error("SSE response has no body");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventName: string | null = null;
  let dataLines: string[] = [];

  const flush = () => {
    if (dataLines.length === 0 && !eventName) return;
    opts.onEvent({ event: eventName, data: dataLines.join("\n") });
    eventName = null;
    dataLines = [];
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE events are terminated by a blank line (\n\n). Split carefully so
    // we don't lose a partial line at the tail of the current chunk.
    let idx;
    while ((idx = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, idx).replace(/\r$/, "");
      buffer = buffer.slice(idx + 1);

      if (line === "") {
        flush();
        continue;
      }
      if (line.startsWith(":")) continue; // comment / keepalive
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
      // Other fields (id:, retry:) ignored — server doesn't use them.
    }
  }

  flush();
  return res;
}
