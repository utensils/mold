import { FALLBACK_VIDEO_FPS } from "@ui/lib/duration";
import { maxFramesForFamilyAtFps } from "./videoBudget";
import {
  MINIMAX_H3_FIXED_FPS,
  MINIMAX_H3_FRAME_OFFSET,
  MINIMAX_H3_FRAME_STEP,
  MINIMAX_H3_MIN_FRAMES,
} from "./minimaxH3Authoring";
import {
  decideGenerateRequestRouting,
  textOnlyAutoChainSingleClipCeiling,
  type GenerateRoutingRequest,
} from "./chainRouting";

/** Frame rate used when nothing else is known — older servers and image
 * models omit the additive `/api/models.default_fps`. */
export const DEFAULT_VIDEO_FPS = FALLBACK_VIDEO_FPS;

/**
 * Frame rate for a selected video model: the model's own server-advertised
 * default (`/api/models.default_fps` — LTX-Video ships 30, LTX-2 24), then
 * whatever the form already holds, then the 24-fps fallback.
 *
 * Every surface applies this on model selection exactly as it applies
 * `default_steps` / `default_guidance`, so the Advanced video summary and the
 * submitted request agree with the model.
 */
export function defaultVideoFps(
  model: { default_fps?: number | null } | null | undefined,
  current?: number | null,
): number {
  return model?.default_fps ?? current ?? DEFAULT_VIDEO_FPS;
}

/** Additive `/api/models` fields that describe a model's requestable video grid. */
export interface VideoFrameContract {
  name?: string | null | undefined;
  family?: string | null | undefined;
  source_image?: string | null | undefined;
  default_frames?: number | null;
  default_fps?: number | null;
  min_frames?: number | null;
  max_frames?: number | null;
  max_runtime_seconds?: number | null;
  max_frames_absolute?: number | null;
  frame_step?: number | null;
  frame_offset?: number | null;
}

export interface VideoGenerationMark {
  frames: number;
  generations: number;
  label: string;
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
  // A tier that hands nothing across a clip boundary has no automatic
  // sequence to reach for, so its clip size is the real UI ceiling — the host's
  // advertised `max_frames` is the family's 257, which submit now refuses.
  // Without this the Duration slider ran to 257 and, because a refused count
  // has no generation count of its own, collapsed every notch into a single
  // "1×" mark reading "257 frames · 16 fps · 16.1s · 1 generation" — an
  // advertised single generation that admission answers 422 to.
  const clipCeiling = textOnlyAutoChainSingleClipCeiling(
    model?.family,
    model?.name ?? "",
    model?.source_image,
    model?.default_frames,
  );
  if (clipCeiling !== null) cap = Math.min(cap, clipCeiling);
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
  if (
    !Number.isInteger(frames) ||
    rounded < offset ||
    (rounded - offset) % step !== 0
  ) {
    const requested = Number.isFinite(frames)
      ? frames
      : (model?.default_frames ?? minimum);
    const down = snapVideoFrames(requested, model, "down");
    const up = down >= requested ? down : down + step;
    const suggestion = down === up ? String(down) : `${down} or ${up}`;
    return `Frames must be ${videoFrameGridLabel(model)} — try ${suggestion}.`;
  }
  return null;
}

/**
 * Keep an authored duration when it is valid for the newly selected model,
 * otherwise enter that model on its advertised default. This is deliberately
 * different from snapping to the nearest grid point: a carried H3 value such
 * as 124 would render as 125 in Wan's slider while the request still contained
 * the rejected 124. Model selection is the authority boundary, so an invalid
 * carried value yields to the target model's own measured default instead.
 */
export function videoFramesForModelSelection(
  frames: number | null | undefined,
  model?: VideoFrameContract | null,
): number {
  const fallback = model?.default_frames ?? frames ?? 25;
  const rate = fixedVideoFps(model) ?? model?.default_fps ?? 24;
  const normalizedDefault = clampVideoFrames(fallback, rate, model);

  // Required-source checkpoints are image-to-video models. Selecting one is
  // a fresh default boundary: carrying a longer duration from the previous
  // checkpoint can silently enter automatic sequencing instead of the model's
  // advertised one-generation clip.
  if (model?.source_image === "required") return normalizedDefault;

  if (frames == null || videoFrameGridError(frames, model)) {
    return normalizedDefault;
  }
  // A carried count the TARGET model refuses is not a duration it can be
  // entered on. 97 is on wan's `4k+1` grid, so it used to survive selection
  // onto an A14B text-to-video tier whose clip is 73 — and that tier refuses
  // an automatic split, so Generate came up disabled with the refusal as its
  // reason before the user touched anything.
  //
  // Asked of the routing authority rather than keyed on the contract string:
  // every LTX-Video tier also advertises `unsupported` and that family IS
  // auto-chained, so a string test would shorten a carried duration there for
  // no reason. An above-ceiling count on a chain-capable family stays exactly
  // as authored — that is the documented way to ask for a long video.
  if (videoGenerationCount(frames, rate, model) === null) {
    return normalizedDefault;
  }
  return frames;
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

/** Number of concrete model invocations used by the ordinary One shot route.
 * This delegates to the same shared routing authority used at submit time so
 * family clip sizes and handoff overlap cannot drift in the UI. */
export function videoGenerationCount(
  frames: number,
  fps: number,
  model?: VideoFrameContract | null,
  routingRequest: Partial<GenerateRoutingRequest> = {},
): number | null {
  const decision = decideGenerateRequestRouting(
    {
      ...routingRequest,
      frames,
      fps,
      model: model?.name ?? routingRequest.model ?? "",
    },
    model?.family,
    model,
  );
  // A REFUSED duration is not a number of generations. Reporting 1 for it let
  // the slider advertise "1 generation" for a frame count the same routing
  // authority was about to reject at submit — the two answers came from one
  // call and still disagreed. `null` is the honest third answer.
  if (decision.kind === "reject") return null;
  return decision.kind === "chain" ? decision.stageCount : 1;
}

/** Natural slider stops: the longest valid duration delivered by each model
 * invocation count. A browser range may snap to the datalist values, while
 * the rendered notches remain an exact visual map even where native snapping
 * is not available. */
export function videoGenerationMarks(
  fps: number,
  model?: VideoFrameContract | null,
  routingRequest: Partial<GenerateRoutingRequest> = {},
): VideoGenerationMark[] {
  const minimum = minVideoFrames(model);
  const maximum = maxVideoFrames(model, fps);
  const step = videoFrameStep(model);
  const marks = new Map<number, number>();
  for (let frames = minimum; frames <= maximum; frames += step) {
    const generations = videoGenerationCount(
      frames,
      fps,
      model,
      routingRequest,
    );
    // A refused stop is not a notch. With the ceiling above this is already
    // unreachable for the tier that motivated it; skipping is what keeps a
    // future refusal from inventing a mark.
    if (generations === null) continue;
    marks.set(generations, frames);
  }
  return [...marks]
    .sort(([left], [right]) => left - right)
    .map(([generations, frames]) => ({
      frames,
      generations,
      label: `${generations}×`,
    }));
}
