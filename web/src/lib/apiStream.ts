import { streamSse } from "./sse";

export type StreamError =
  | { kind: "http"; status: number; retryAfter?: number; body: string }
  | { kind: "network"; message: string };

export interface PostSseJsonHandlers<TProgress, TComplete> {
  onProgress: (evt: TProgress) => void;
  onComplete: (evt: TComplete) => void;
  onError: (err: StreamError) => void;
}

export interface PostSseJsonOptions<TBody, TProgress, TComplete> {
  url: string;
  body: TBody;
  signal?: AbortSignal;
  /** Extra request headers (the routed host's `x-api-key`). */
  headers?: Record<string, string>;
  handlers: PostSseJsonHandlers<TProgress, TComplete>;
  silentCloseMessage: string;
}

function retryAfterSeconds(res: Response): number | undefined {
  const raw = res.headers.get("Retry-After");
  if (!raw) return undefined;
  const parsed = Number.parseFloat(raw);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export async function postSseJsonStream<TBody, TProgress, TComplete>(
  opts: PostSseJsonOptions<TBody, TProgress, TComplete>,
): Promise<void> {
  let terminal = false;
  try {
    await streamSse({
      url: opts.url,
      body: opts.body,
      signal: opts.signal,
      headers: opts.headers,
      onEvent: (evt) => {
        if (!evt.data) return;
        let parsed: unknown;
        try {
          parsed = JSON.parse(evt.data);
        } catch {
          return;
        }
        if (evt.event === "complete") {
          terminal = true;
          opts.handlers.onComplete(parsed as TComplete);
        } else if (evt.event === "error") {
          terminal = true;
          opts.handlers.onError({ kind: "http", status: 0, body: evt.data });
        } else {
          opts.handlers.onProgress(parsed as TProgress);
        }
      },
      onHttpError: (res) => {
        terminal = true;
        const retryAfter = retryAfterSeconds(res);
        res
          .text()
          .then((body) =>
            opts.handlers.onError({
              kind: "http",
              status: res.status,
              retryAfter,
              body,
            }),
          )
          .catch(() =>
            opts.handlers.onError({
              kind: "http",
              status: res.status,
              retryAfter,
              body: "",
            }),
          );
      },
    });
    if (!terminal && !opts.signal?.aborted) {
      opts.handlers.onError({
        kind: "network",
        message: opts.silentCloseMessage,
      });
    }
  } catch (err) {
    if (opts.signal?.aborted || terminal) return;
    opts.handlers.onError({
      kind: "network",
      message: err instanceof Error ? err.message : String(err),
    });
  }
}
