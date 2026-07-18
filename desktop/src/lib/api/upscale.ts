/**
 * Standalone image upscaling over `POST /api/upscale/stream` (SSE) — used by
 * the img2img "upscale then fit" source preprocessing. The streaming route is
 * preferred over plain `/api/upscale` because its `complete` event carries the
 * image as base64 (the JSON route serializes raw bytes as a number array) and
 * it reports auto-pull/stage progress for large first-use downloads.
 *
 * Multi-host: pass `target` to run the upscale on the host the generation
 * will route to — the upscaler model auto-downloads there on first use.
 */
import type { ApiTarget } from "./client";
import { sseStream } from "./sse";

export interface UpscaleImageOptions {
  /** Upscaler model name, e.g. "real-esrgan-x2plus:fp16". */
  model: string;
  /** Input image, base64 (no data-URI prefix). */
  image: string;
  /** Explicit host; defaults to the primary connection. */
  target?: ApiTarget;
  signal?: AbortSignal;
  /** Stage/progress messages for status UI. */
  onProgress?: (message: string) => void;
}

/** Upscale an image, resolving to the upscaled base64 PNG. */
export async function upscaleImage(options: UpscaleImageOptions): Promise<string> {
  let image: string | null = null;
  let failure: Error | null = null;
  // Always drive our own controller so we can cut the stream the moment a
  // terminal frame lands. A server that emits `error` (or even `complete`)
  // without closing the connection would otherwise leave sseStream's promise
  // pending until connection close, hanging the submit path. Forward the
  // caller's signal into it so external cancellation is still honored.
  const abort = new AbortController();
  if (options.signal) {
    if (options.signal.aborted) abort.abort();
    else options.signal.addEventListener("abort", () => abort.abort(), { once: true });
  }

  await sseStream("/api/upscale/stream", {
    method: "POST",
    body: { model: options.model, image: options.image, output_format: "png" },
    signal: abort.signal,
    retry: false,
    ...(options.target ? { target: options.target } : {}),
    onEvent: (event, data) => {
      if (event === "error") {
        let message = data;
        try {
          const parsed = JSON.parse(data) as { error?: string; message?: string };
          message = parsed.error ?? parsed.message ?? data;
        } catch {
          /* non-JSON error body — keep the raw text */
        }
        failure = new Error(message);
        abort.abort();
        return;
      }
      try {
        if (event === "progress") {
          const evt = JSON.parse(data) as { type?: string; name?: string; message?: string };
          const message = evt.message ?? evt.name;
          if (message) options.onProgress?.(message);
        } else if (event === "complete") {
          image = (JSON.parse(data) as { image: string }).image;
          abort.abort();
        }
      } catch {
        /* skip malformed frame */
      }
    },
    onClose: (err) => {
      if (err && !failure) failure = err;
    },
  });

  if (failure) throw failure;
  if (image === null) throw new Error("upscale stream closed before completion");
  return image;
}
