import { ApiError, apiJsonTo, type ApiTarget } from "./client";
import type { QueueEntry } from "./queuePlan";

/** Metadata is the one cross-surface restore authority: Library reuse and
 * queue selection both feed the same complete saved-settings mapper. */
export function settingsRestoreMetadata<T extends object>(
  metadata: T,
  options: { seedPinned?: boolean | null | undefined } = {},
): T {
  return options.seedPinned === false
    ? ({ ...metadata, seed: null } as T)
    : ({ ...metadata } as T);
}

export interface SelectedQueueGeneration<T extends object> {
  metadata: T;
  jobId: string;
  running: boolean;
}

export interface SelectedQueuePreviewSource {
  hostId: string;
  jobId: string;
  running: boolean;
}

export function selectedQueueGeneration<T extends object>(
  entries: readonly QueueEntry[],
  jobId: string,
): SelectedQueueGeneration<T> | null {
  const entry = entries.find((candidate) => candidate.id === jobId);
  if (!entry || typeof entry.metadata !== "object" || entry.metadata === null)
    return null;
  return {
    metadata: settingsRestoreMetadata(entry.metadata as T, {
      seedPinned: entry.seed_pinned,
    }),
    jobId: entry.id,
    running: entry.state === "running",
  };
}

export interface QueueJobPreview {
  image: string;
  step: number;
  total: number;
}

function parseQueueJobPreview(value: unknown): QueueJobPreview | null {
  if (typeof value !== "object" || value === null) return null;
  const row = value as Record<string, unknown>;
  return typeof row.image === "string" &&
    Number.isFinite(row.step) &&
    Number.isFinite(row.total) &&
    (row.total as number) > 0
    ? {
        image: row.image,
        step: row.step as number,
        total: row.total as number,
      }
    : null;
}

/** Poll only the explicitly selected running job. A live job returns null
 * before its first supported denoise step; 404 means the live row is gone.
 * Older hosts simply fall back to settings restoration and the normal canvas. */
export function watchSelectedQueuePreview(
  target: ApiTarget,
  jobId: string,
  onPreview: (preview: QueueJobPreview) => void,
  intervalMs = 750,
  onEnded?: () => void,
): () => void {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | null = null;
  let stopped = false;

  const tick = async () => {
    try {
      const value = await apiJsonTo<unknown>(
        target,
        `/api/queue/${encodeURIComponent(jobId)}/preview`,
        { signal: controller.signal },
      );
      const preview = parseQueueJobPreview(value);
      if (preview && !stopped) {
        onPreview(preview);
      }
    } catch (error) {
      // Selection and full settings restore stay useful on older hosts and
      // through transient network errors. Only server-confirmed disappearance
      // releases the selected canvas.
      if (!stopped && error instanceof ApiError && error.status === 404) {
        stopped = true;
        onEnded?.();
      }
    } finally {
      if (!stopped) timer = setTimeout(tick, intervalMs);
    }
  };
  void tick();

  return () => {
    stopped = true;
    controller.abort();
    if (timer !== null) clearTimeout(timer);
  };
}
