import type { ApiTarget } from "./client";
import {
  watchSelectedQueuePreview,
  type QueueJobProgress,
} from "./generationSelection";

/**
 * Live progress for the prints THIS client submitted.
 *
 * The attached SSE stream used to push `preview` and `progress` frames for a
 * print; the durable path carries neither — its authority is the batch child
 * state, and `GET /api/queue/{id}/preview` is the only producer of a stage,
 * a step count, and (when previews are enabled) a denoise image. Every
 * surface therefore polls that endpoint for each of its own jobs while the
 * child is `running`, exactly as it already does for an inspected queue row,
 * and stops the moment the child leaves that state. One watcher per client
 * job; re-asking for the same server job is a no-op, so a reducer can call
 * `ensure` on every snapshot.
 */
export type OwnPrintPreviewWatch = typeof watchSelectedQueuePreview;

export class OwnPrintPreviewWatchers {
  private readonly active = new Map<
    string,
    { jobId: string; stop: () => void }
  >();

  constructor(
    private readonly watch: OwnPrintPreviewWatch = watchSelectedQueuePreview,
    private readonly intervalMs = 750,
  ) {}

  /** Poll `jobId` on `target` for the client job `key`; idempotent per job id. */
  ensure(
    key: string,
    target: ApiTarget,
    jobId: string,
    onPreview: (progress: QueueJobProgress) => void,
  ): void {
    const current = this.active.get(key);
    if (current?.jobId === jobId) return;
    current?.stop();
    const stop = this.watch(target, jobId, onPreview, this.intervalMs, () => {
      // The host says the live row is gone: nothing more to poll.
      if (this.active.get(key)?.jobId === jobId) this.active.delete(key);
    });
    this.active.set(key, { jobId, stop });
  }

  stop(key: string): void {
    const current = this.active.get(key);
    if (!current) return;
    this.active.delete(key);
    current.stop();
  }

  stopAll(): void {
    for (const key of [...this.active.keys()]) this.stop(key);
  }

  has(key: string): boolean {
    return this.active.has(key);
  }
}

/**
 * The data URL a surface renders a polled denoise preview from, or `null`
 * when this host is rendering with previews disabled — the step counter
 * beside it still works.
 */
export function previewDataUrl(progress: QueueJobProgress): string | null {
  return progress.preview_image
    ? `data:image/png;base64,${progress.preview_image}`
    : null;
}
