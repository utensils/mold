import { maxFramesForFamilyAtFps } from "./videoBudget";

/** Additive `/api/models` fields that describe a model's requestable video grid. */
export interface VideoFrameContract {
  family?: string | null;
  default_frames?: number | null;
  default_fps?: number | null;
  max_frames?: number | null;
  max_runtime_seconds?: number | null;
  max_frames_absolute?: number | null;
  frame_step?: number | null;
}

const LEGACY_FRAME_STEP = 8;
const LEGACY_MAX_FRAMES = 257;

export function videoFrameStep(model?: VideoFrameContract | null): number {
  const step = Math.round(model?.frame_step ?? LEGACY_FRAME_STEP);
  return step > 0 ? step : LEGACY_FRAME_STEP;
}

/** Snap a frame count onto the server's `k * step + 1` request grid. */
export function snapVideoFrames(
  frames: number,
  model?: VideoFrameContract | null,
  direction: "nearest" | "down" = "nearest",
): number {
  const step = videoFrameStep(model);
  const scaled = (Math.max(1, frames) - 1) / step;
  const grid = direction === "down" ? Math.floor(scaled) : Math.round(scaled);
  return Math.max(1, grid * step + 1);
}

/**
 * Requestable single-shot ceiling at the currently selected FPS.
 *
 * LTX-2 advertises a duration ceiling, so its frame limit moves with FPS.
 * Older servers only advertise `max_frames`; the conservative legacy fallback
 * keeps those clients useful without inventing a larger model limit.
 */
export function maxVideoFrames(
  model: VideoFrameContract | null | undefined,
  fps: number,
): number {
  const rate = Math.max(1, Math.round(fps) || model?.default_fps || 24);
  let cap: number;
  if (model?.max_runtime_seconds) {
    cap = model.max_runtime_seconds * rate + 4;
    if (model.max_frames_absolute)
      cap = Math.min(cap, model.max_frames_absolute);
  } else {
    cap =
      model?.max_frames ??
      maxFramesForFamilyAtFps(model?.family, rate) ??
      LEGACY_MAX_FRAMES;
  }
  return snapVideoFrames(cap, model, "down");
}

export function minVideoFrames(): number {
  return 1;
}

export function clampVideoFrames(
  frames: number,
  fps: number,
  model?: VideoFrameContract | null,
): number {
  return Math.min(
    maxVideoFrames(model, fps),
    Math.max(minVideoFrames(), snapVideoFrames(frames, model)),
  );
}

export function videoDurationSeconds(frames: number, fps: number): number {
  return Math.max(1, frames) / Math.max(1, fps);
}

export function framesForVideoDuration(
  seconds: number,
  fps: number,
  model?: VideoFrameContract | null,
): number {
  return clampVideoFrames(seconds * Math.max(1, fps), fps, model);
}

export function formatVideoDuration(frames: number, fps: number): string {
  const seconds = videoDurationSeconds(frames, fps);
  if (seconds >= 10 && Math.abs(seconds - Math.round(seconds)) < 0.05) {
    return `${Math.round(seconds)}s`;
  }
  return `${seconds.toFixed(1)}s`;
}
