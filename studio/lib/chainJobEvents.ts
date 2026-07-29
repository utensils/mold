/**
 * Pure chain-job SSE event reducer, shared by desktop, web, and iPhone.
 * First frame is a `snapshot` (full detail); deltas advance the active
 * stage's state and denoise progress. Ported from the desktop store so all
 * three surfaces reduce `/api/chain-jobs/:id/events` identically.
 */

import type {
  ChainJobDetail,
  ChainJobEvent,
  ChainJobStageDetail,
  ChainStageState,
} from "./api/chainTypes";

/** Live view of the watched job: detail + per-stage denoise progress. */
export interface ChainJobLive {
  detail: ChainJobDetail | null;
  progress: Record<number, { step: number; total: number }>;
  activeStage: number | null;
}

export function emptyChainJobLive(): ChainJobLive {
  return { detail: null, progress: {}, activeStage: null };
}

function firstRunningStage(job: ChainJobDetail): number | null {
  const running = job.stages.find((s) => s.state === "running");
  return running ? running.idx : null;
}

function withStageState(
  detail: ChainJobDetail | null,
  idx: number,
  state: ChainStageState,
  availability?: {
    hasPreview: boolean | undefined;
    hasMedia: boolean | undefined;
    cacheReady: boolean | undefined;
  },
): ChainJobDetail | null {
  if (!detail) return detail;
  const stages = detail.stages.map((s): ChainJobStageDetail =>
    s.idx === idx
      ? {
          ...s,
          state,
          has_preview: availability?.hasPreview ?? s.has_preview,
          has_media: availability?.hasMedia ?? s.has_media,
          cache_ready: availability?.cacheReady ?? s.cache_ready,
        }
      : s,
  );
  return { ...detail, stages };
}

export function applyChainJobEvent(
  state: ChainJobLive,
  ev: ChainJobEvent,
): ChainJobLive {
  switch (ev.type) {
    case "snapshot":
      return {
        detail: ev.job,
        progress: {},
        activeStage: firstRunningStage(ev.job),
      };
    case "stage_start":
      return {
        ...state,
        activeStage: ev.stage_idx,
        detail: withStageState(state.detail, ev.stage_idx, "running"),
      };
    case "denoise_step":
      return {
        ...state,
        progress: {
          ...state.progress,
          [ev.stage_idx]: { step: ev.step, total: ev.total },
        },
      };
    case "stage_done":
      return {
        ...state,
        activeStage:
          state.activeStage === ev.stage_idx ? null : state.activeStage,
        detail: withStageState(state.detail, ev.stage_idx, "completed", {
          hasPreview: ev.has_preview,
          hasMedia: ev.has_media,
          cacheReady: ev.cache_ready,
        }),
      };
    case "state_changed":
      return {
        ...state,
        detail: state.detail
          ? { ...state.detail, state: ev.state, error: ev.error ?? null }
          : null,
      };
    default:
      return state;
  }
}
