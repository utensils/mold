import { maxFramesForFamilyAtFps } from "./videoBudget";
import {
  MINIMAX_H3_FIXED_FPS,
  MINIMAX_H3_FRAME_OFFSET,
  MINIMAX_H3_FRAME_STEP,
  MINIMAX_H3_MIN_FRAMES,
} from "./minimaxH3Authoring";

/** Additive `/api/models` fields that describe a model's requestable video grid. */
export interface VideoFrameContract {
  family?: string | null;
  default_frames?: number | null;
  default_fps?: number | null;
  min_frames?: number | null;
  max_frames?: number | null;
  max_runtime_seconds?: number | null;
  max_frames_absolute?: number | null;
  frame_step?: number | null;
  frame_offset?: number | null;
}

const LEGACY_FRAME_STEP = 8;
const LEGACY_MAX_FRAMES = 257;

/**
 * Frame grid per family, mirroring `frame_step_for_family` in
 * `crates/mold-core/src/validation.rs`. Valid counts are
 * `k * step + videoFrameOffset(model)`.
 *
 * This is the fallback for a model row that names its family but not its
 * `frame_step`. It matters because the legacy default is 8 and Wan's grid is
 * 4: without the family arm a Wan row snaps 81 to 81 but 49 to 41, and any
 * count the user lands on that is `4k+1` but not `8k+1` is rewritten to a
 * neighbouring value or rejected outright by the server's validator.
 */
const FAMILY_FRAME_STEP: ReadonlyMap<string, number> = new Map([
  ["ltx2", 8],
  ["ltx-2", 8],
  ["ltx-video", 8],
  ["wan", 4],
  ["minimax-h3", MINIMAX_H3_FRAME_STEP],
  ["minimax_h3", MINIMAX_H3_FRAME_STEP],
  ["minimaxh3", MINIMAX_H3_FRAME_STEP],
]);

const FAMILY_FRAME_OFFSET: ReadonlyMap<string, number> = new Map([
  ["minimax-h3", MINIMAX_H3_FRAME_OFFSET],
  ["minimax_h3", MINIMAX_H3_FRAME_OFFSET],
  ["minimaxh3", MINIMAX_H3_FRAME_OFFSET],
]);

const FAMILY_MIN_FRAMES: ReadonlyMap<string, number> = new Map([
  ["minimax-h3", MINIMAX_H3_MIN_FRAMES],
  ["minimax_h3", MINIMAX_H3_MIN_FRAMES],
  ["minimaxh3", MINIMAX_H3_MIN_FRAMES],
]);

const FAMILY_FIXED_FPS: ReadonlyMap<string, number> = new Map([
  ["minimax-h3", MINIMAX_H3_FIXED_FPS],
  ["minimax_h3", MINIMAX_H3_FIXED_FPS],
  ["minimaxh3", MINIMAX_H3_FIXED_FPS],
]);

export function videoFrameStep(model?: VideoFrameContract | null): number {
  const advertised = model?.frame_step;
  if (advertised != null) {
    const step = Math.round(advertised);
    if (step > 0) return step;
  }
  const family = (model?.family ?? "").trim().toLowerCase();
  return FAMILY_FRAME_STEP.get(family) ?? LEGACY_FRAME_STEP;
}

export function videoFrameOffset(model?: VideoFrameContract | null): number {
  const advertised = model?.frame_offset;
  if (advertised != null) {
    const offset = Math.round(advertised);
    if (offset > 0) return offset;
  }
  const family = (model?.family ?? "").trim().toLowerCase();
  return FAMILY_FRAME_OFFSET.get(family) ?? 1;
}

/** Snap a frame count onto the server's `k * step + offset` request grid. */
export function snapVideoFrames(
  frames: number,
  model?: VideoFrameContract | null,
  direction: "nearest" | "down" = "nearest",
): number {
  const step = videoFrameStep(model);
  const offset = videoFrameOffset(model);
  const scaled = (Math.max(offset, frames) - offset) / step;
  const grid = direction === "down" ? Math.floor(scaled) : Math.round(scaled);
  return Math.max(offset, grid * step + offset);
}

/** Validate a frame count against the selected model's advertised grid. */
export function videoFramesError(
  frames: number,
  model?: VideoFrameContract | null,
): string | null {
  const step = videoFrameStep(model);
  if (Number.isInteger(frames) && frames > 0 && (frames - 1) % step === 0) {
    return null;
  }
  const requested = Number.isFinite(frames)
    ? frames
    : (model?.default_frames ?? 1);
  const down = snapVideoFrames(requested, model, "down");
  const up = down >= requested ? down : down + step;
  const suggestion = down === up ? String(down) : `${down} or ${up}`;
  return `Frames must be ${step}n+1 — try ${suggestion}.`;
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

export function minVideoFrames(model?: VideoFrameContract | null): number {
  const advertised = model?.min_frames;
  if (advertised != null) {
    const minimum = Math.round(advertised);
    if (minimum > 0) return minimum;
  }
  const family = (model?.family ?? "").trim().toLowerCase();
  return FAMILY_MIN_FRAMES.get(family) ?? 1;
}

export function fixedVideoFps(
  model?: VideoFrameContract | null,
): number | null {
  const family = (model?.family ?? "").trim().toLowerCase();
  return FAMILY_FIXED_FPS.get(family) ?? null;
}

export function videoFrameGridLabel(model?: VideoFrameContract | null): string {
  return `${videoFrameStep(model)}n+${videoFrameOffset(model)}`;
}

export function videoFramesError(
  frames: number,
  model?: VideoFrameContract | null,
): string | null {
  const rounded = Math.round(frames);
  const minimum = minVideoFrames(model);
  const maximum = maxVideoFrames(
    model,
    fixedVideoFps(model) ?? model?.default_fps ?? 24,
  );
  if (rounded < minimum || rounded > maximum) {
    return `Frames must be between ${minimum} and ${maximum}.`;
  }
  return videoFrameGridError(frames, model);
}

/** Validate only the model's discrete grid, preserving above-ceiling exact
 * values that a chain-capable family routes into an automatic sequence. */
export function videoFrameGridError(
  frames: number,
  model?: VideoFrameContract | null,
): string | null {
  const rounded = Math.round(frames);
  const minimum = minVideoFrames(model);
  const step = videoFrameStep(model);
  const offset = videoFrameOffset(model);
  if (rounded < minimum) return `Frames must be at least ${minimum}.`;
  if (rounded < offset || (rounded - offset) % step !== 0) {
    return `Frames must follow the ${videoFrameGridLabel(model)} grid.`;
  }
  return null;
}

export function clampVideoFrames(
  frames: number,
  fps: number,
  model?: VideoFrameContract | null,
): number {
  return Math.min(
    maxVideoFrames(model, fps),
    Math.max(minVideoFrames(model), snapVideoFrames(frames, model)),
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
  if (seconds < 0.1) return `${seconds.toFixed(2)}s`;
  return `${seconds.toFixed(1)}s`;
}
