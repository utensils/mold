/**
 * One translation from the durable chain-job event stream into the progress
 * frames the Create activity rail already reduces.
 *
 * `/api/generate/chain/stream` is gone: an auto-chained sequence is created
 * through `POST /api/chain-jobs` and followed on
 * `GET /api/chain-jobs/:id/events` like any other chain job. The two event
 * vocabularies differ in three ways, and each is handled here rather than at
 * a call site so desktop and web cannot drift:
 *
 * - the durable stream opens with a full `snapshot` where the shim sent a
 *   synthesized `chain_start`;
 * - `stage_done` carries no `frames_emitted`, so it is read back off the
 *   snapshot the reducer maintains;
 * - completion is `finalized { output }` — a saved FILENAME, never inline
 *   bytes — so the caller fetches the media from that machine's gallery
 *   exactly as a durable print does.
 */

import {
  applyChainJobEvent,
  emptyChainJobLive,
  type ChainJobLive,
} from "./chainJobEvents";
import type { ChainJobEvent, ChainJobState } from "./api/chainTypes";

/**
 * Structurally the `ChainProgressEvent` union both shells already own, so a
 * frame can be handed straight to their existing reducers. Deliberately
 * redeclared here rather than imported: `studio` must not depend on either
 * application shell.
 */
export type ChainJobProgressFrame =
  | { type: "chain_start"; stage_count: number; estimated_total_frames: number }
  | { type: "stage_start"; stage_idx: number }
  | { type: "denoise_step"; stage_idx: number; step: number; total: number }
  | { type: "stage_done"; stage_idx: number; frames_emitted: number }
  | { type: "stitching"; total_frames: number };

export interface ChainJobFinalized {
  /** The gallery filename the stitched print was saved under. */
  output: string | null;
  take: number | null;
}

export interface ChainJobTerminal {
  state: ChainJobState;
  error: string | null;
}

export interface ChainJobFrameResult {
  live: ChainJobLive;
  /** Zero or one progress frame; `yielded` produces none. */
  progress: ChainJobProgressFrame[];
  finalized: ChainJobFinalized | null;
  terminal: ChainJobTerminal | null;
}

export { emptyChainJobLive };

/** The clip lengths the job's own script declares, or 0 when it declares none. */
export function estimatedChainFrames(live: ChainJobLive): number {
  const stages = live.detail?.script?.stages;
  if (!Array.isArray(stages)) return 0;
  return stages.reduce(
    (total, stage) =>
      total + (typeof stage.frames === "number" ? stage.frames : 0),
    0,
  );
}

function framesEmitted(live: ChainJobLive, stageIdx: number): number {
  const stage = live.detail?.stages.find(
    (candidate) => candidate.idx === stageIdx,
  );
  return typeof stage?.frames_emitted === "number" ? stage.frames_emitted : 0;
}

function isTerminal(state: ChainJobState): boolean {
  return state === "completed" || state === "failed" || state === "cancelled";
}

/**
 * Advance the live snapshot and say what the surface should do with the
 * frame. The live value is always returned so the caller keeps one authority
 * for the job rather than a second parallel reduction.
 */
function snapshotFinalized(job: {
  finalizes?: ReadonlyArray<{
    gallery_filename?: string | null;
    take?: number;
  }> | null;
}): ChainJobFinalized | null {
  const last = job.finalizes?.at(-1);
  if (!last?.gallery_filename) return null;
  return { output: last.gallery_filename, take: last.take ?? null };
}

export function reduceChainJobFrame(
  previous: ChainJobLive,
  event: ChainJobEvent,
): ChainJobFrameResult {
  const live = applyChainJobEvent(previous, event);
  const base: ChainJobFrameResult = {
    live,
    progress: [],
    finalized: null,
    terminal: null,
  };
  switch (event.type) {
    case "snapshot":
      return {
        ...base,
        progress: [
          {
            type: "chain_start",
            stage_count: event.job.stage_count,
            estimated_total_frames: estimatedChainFrames(live),
          },
        ],
        // A job that is already terminal when we attach still has to settle,
        // and its print is then in the manifest's last finalize record — the
        // only place the gallery filename exists after the stream is gone.
        finalized: isTerminal(event.job.state)
          ? snapshotFinalized(event.job)
          : live.finalized,
        terminal: isTerminal(event.job.state)
          ? { state: event.job.state, error: event.job.error ?? null }
          : null,
      };
    case "stage_start":
      return {
        ...base,
        progress: [{ type: "stage_start", stage_idx: event.stage_idx }],
      };
    case "denoise_step":
      return {
        ...base,
        progress: [
          {
            type: "denoise_step",
            stage_idx: event.stage_idx,
            step: event.step,
            total: event.total,
          },
        ],
      };
    case "stage_done":
      return {
        ...base,
        progress: [
          {
            type: "stage_done",
            stage_idx: event.stage_idx,
            frames_emitted: framesEmitted(live, event.stage_idx),
          },
        ],
      };
    case "finalizing":
      return {
        ...base,
        progress: [
          {
            type: "stitching",
            total_frames: event.total_frames ?? estimatedChainFrames(live),
          },
        ],
      };
    case "finalized":
      // `output` is the job-relative MP4 artifact amend/retake decode; the
      // print a client fetches is the gallery filename, in the requested
      // format. A host with no gallery output publishes nothing to fetch.
      return {
        ...base,
        finalized: {
          output: event.gallery_filename ?? null,
          take: event.take ?? null,
        },
      };
    case "state_changed":
      return {
        ...base,
        terminal: isTerminal(event.state)
          ? { state: event.state, error: event.error ?? null }
          : null,
      };
    default:
      // `yielded` and any frame a newer server adds carry no surface meaning.
      return base;
  }
}
