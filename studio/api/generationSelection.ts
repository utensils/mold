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

/**
 * The host's folded progress snapshot for one live queue row.
 *
 * A denoise image is only one of the things it can carry: a host running with
 * `MOLD_STEP_PREVIEW=0` reports steps and stage with no image at all, which is
 * why every field is independently optional.
 */
export interface QueueJobProgress {
  step: number | null;
  total: number | null;
  stage: string | null;
  queue_position: number | null;
  preview_image: string | null;
}

function finiteOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function parseQueueJobProgress(value: unknown): QueueJobProgress | null {
  if (typeof value !== "object" || value === null) return null;
  const row = value as Record<string, unknown>;
  const total = finiteOrNull(row.total);
  const progress: QueueJobProgress = {
    step: finiteOrNull(row.step),
    total: total !== null && total > 0 ? total : null,
    stage: typeof row.stage === "string" && row.stage ? row.stage : null,
    queue_position: finiteOrNull(row.queue_position),
    preview_image:
      typeof row.preview_image === "string" && row.preview_image
        ? row.preview_image
        : null,
  };
  // Nothing worth reporting yet: the row exists but has produced no progress.
  return progress.step === null &&
    progress.stage === null &&
    progress.queue_position === null &&
    progress.preview_image === null
    ? null
    : progress;
}

/** Poll only the explicitly selected running job. A live job returns null
 * before it reports any progress; 404 means the live row is gone. */
export function watchSelectedQueuePreview(
  target: ApiTarget,
  jobId: string,
  onPreview: (preview: QueueJobProgress) => void,
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
      const preview = parseQueueJobProgress(value);
      if (preview && !stopped) {
        onPreview(preview);
      }
    } catch (error) {
      // Selection and full settings restore stay useful through transient
      // network errors. Only server-confirmed disappearance releases the
      // selected canvas.
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
