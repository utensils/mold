/**
 * How a durable duration parent reads as an ordinary Queue row.
 *
 * One wording on every surface. `docs/contracts/video-job-rows.json` is the
 * cross-language fixture; `mold_core::video_job_row` (Rust) and MoldClient's
 * `VideoJobRow.swift` are pinned to the same file.
 */
import type { ActiveWorkItem } from "../api/activity";
import type { VideoJob, VideoJobState } from "./api/videoJobTypes";

export type VideoJobRowSymbol =
  | "waiting"
  | "rendering"
  | "joining"
  | "stopping"
  | "paused"
  | "interrupted"
  | "failed"
  | "stopped"
  | "done";

export interface VideoJobRow {
  caption: string;
  detail: string;
  symbol: VideoJobRowSymbol;
  /** Completed fraction for a determinate meter; only multi-clip work in
   * progress has one. */
  progress: number | null;
}

/** The facts a row is drawn from. Counters are the wire's decimal strings.
 * `current` is false when only a last cached snapshot can be shown. */
export interface VideoJobRowInput {
  state: VideoJobState;
  /** Absent on an older server; an unknown value reads as rendering. */
  progress_phase?: string | null;
  completed_clips: string;
  clip_count: string;
  total_duration_ms: string;
  can_cancel: boolean;
  current: boolean;
}

function count(value: string | number | null | undefined): bigint {
  const text = String(value ?? "0");
  return /^\d+$/.test(text) ? BigInt(text) : 0n;
}

/** `14000` → `14 s`, `14375` → `14.4 s` (one decimal, trailing zero dropped). */
function seconds(milliseconds: bigint): string {
  const tenths = (milliseconds + 50n) / 100n;
  return tenths % 10n === 0n
    ? `${tenths / 10n} s`
    : `${tenths / 10n}.${tenths % 10n} s`;
}

function captionOf(
  input: Omit<VideoJobRowInput, "total_duration_ms" | "current">,
): Omit<VideoJobRow, "detail"> {
  const clips = count(input.clip_count) > 0n ? count(input.clip_count) : 1n;
  const completed = count(input.completed_clips);
  const done = completed < clips ? completed : clips;
  const multi = clips > 1n;
  const tally = (verb: string) =>
    multi ? `${verb} · ${done} of ${clips} clips done` : verb;
  const meter = multi ? Number(done) / Number(clips) : null;
  switch (input.state) {
    case "queued":
      return { caption: "Waiting", symbol: "waiting", progress: null };
    case "running":
      if (!input.can_cancel)
        return { caption: "Stopping", symbol: "stopping", progress: null };
      if (input.progress_phase === "waiting" && done === 0n)
        return { caption: "Waiting", symbol: "waiting", progress: null };
      if (input.progress_phase === "joining")
        return {
          caption: multi ? "Joining clips" : "Finishing",
          symbol: "joining",
          progress: meter,
        };
      // Rendering, a waiting later clip, or an older server's bare `running`
      // all read as the movie being rendered.
      return multi
        ? {
            caption: `Rendering clip ${done + 1n < clips ? done + 1n : clips} of ${clips}`,
            symbol: "rendering",
            progress: meter,
          }
        : { caption: "Rendering", symbol: "rendering", progress: null };
    case "paused":
      return { caption: tally("Paused"), symbol: "paused", progress: null };
    case "interrupted":
      return {
        caption: tally("Interrupted"),
        symbol: "interrupted",
        progress: null,
      };
    case "failed":
      return { caption: "Failed", symbol: "failed", progress: null };
    case "cancelled":
      return { caption: "Stopped", symbol: "stopped", progress: null };
    case "completed":
      return { caption: "Done", symbol: "done", progress: null };
  }
}

export function videoJobRow(input: VideoJobRowInput): VideoJobRow {
  const row = captionOf(input);
  const clips = count(input.clip_count) > 0n ? count(input.clip_count) : 1n;
  const length = seconds(count(input.total_duration_ms));
  return {
    caption: input.current ? row.caption : `Last known: ${row.caption}`,
    detail: clips > 1n ? `${length} video · ${clips} clips` : `${length} video`,
    symbol: row.symbol,
    progress: row.progress,
  };
}

export function videoJobRowFromJob(job: VideoJob, current = true): VideoJobRow {
  return videoJobRow({
    state: job.state,
    progress_phase: job.progress_phase ?? null,
    completed_clips: job.completed_clips,
    clip_count: job.plan.clip_count,
    total_duration_ms: job.plan.total_duration_ms,
    can_cancel: job.can_cancel,
    current,
  });
}

const SETTLED_STATES: Record<string, VideoJobState> = {
  paused: "paused",
  interrupted: "interrupted",
  failed: "failed",
  cancelled: "cancelled",
  completed: "completed",
};

/**
 * The same caption from an `/api/activity` `video_job` item. The host reports
 * `queued` while the parent's clip still waits for a GPU, `cancelling` once a
 * Stop is accepted, and `current`/`total` as completed clips of the plan. The
 * activity item carries no duration, so it has no detail line.
 */
export function videoJobActivityRow(
  item: ActiveWorkItem,
): Omit<VideoJobRow, "detail"> {
  const completed = item.current ?? 0;
  const clips = item.total ?? 1;
  const settled = SETTLED_STATES[item.phase];
  return captionOf({
    state: settled ?? "running",
    progress_phase:
      item.phase === "queued"
        ? "waiting"
        : completed >= clips
          ? "joining"
          : "rendering",
    completed_clips: String(completed),
    clip_count: String(clips),
    can_cancel: item.phase !== "cancelling",
  });
}

/** A video parent that is still occupying (or waiting for) a GPU. Paused,
 * failed and interrupted parents are recovery work, never live work. */
export function isLiveVideoJobActivity(item: ActiveWorkItem): boolean {
  return (
    item.kind === "video_job" &&
    ["queued", "running", "cancelling"].includes(item.phase)
  );
}

/** Live video parents split the way a queue summary counts them: a parent
 * whose clip still waits for a GPU is waiting, never "being made". */
export function liveVideoJobCounts(items: readonly ActiveWorkItem[]): {
  running: number;
  waiting: number;
} {
  let running = 0,
    waiting = 0;
  for (const item of items) {
    if (!isLiveVideoJobActivity(item)) continue;
    if (item.phase === "queued") waiting += 1;
    else running += 1;
  }
  return { running, waiting };
}

/** A saved `VideoJob` as the `/api/activity` item the host would report for
 * it, so a saved parent renders through the same ordinary row as live work. */
export function videoJobActivityItem(job: VideoJob): ActiveWorkItem {
  const phase =
    job.state === "running"
      ? !job.can_cancel
        ? "cancelling"
        : job.progress_phase === "waiting"
          ? "queued"
          : "running"
      : job.state;
  return {
    id: job.id,
    kind: "video_job",
    execution: "video_job",
    phase,
    model: job.plan?.model ?? null,
    created_at_unix_ms: job.created_at_ms,
    updated_at_unix_ms: job.updated_at_ms,
    current: Number(count(job.completed_clips)),
    total: Number(count(job.plan?.clip_count)),
    can_cancel: job.can_cancel === true,
  };
}
