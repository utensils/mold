import { describe, expect, it } from "vitest";
import {
  emptyChainJobLive,
  estimatedChainFrames,
  reduceChainJobFrame,
  type ChainJobFrameResult,
  type ChainJobProgressFrame,
} from "./chainJobProgress";
import type { ChainJobDetail, ChainJobEvent } from "./api/chainTypes";

function detail(overrides: Partial<ChainJobDetail> = {}): ChainJobDetail {
  return {
    id: "job-1",
    state: "running",
    model: "ltx-2-19b-distilled:fp8",
    stage_count: 2,
    current_stage: 0,
    created_at_unix_ms: 1,
    updated_at_unix_ms: 2,
    error: null,
    ephemeral: true,
    stages: [
      { idx: 0, state: "pending", seed: "42", frames_emitted: null },
      { idx: 1, state: "pending", seed: "43", frames_emitted: null },
    ],
    script: {
      chain: { model: "ltx-2-19b-distilled:fp8" },
      stages: [
        { prompt: "one", frames: 97 },
        { prompt: "two", frames: 97 },
      ],
    },
    ...overrides,
  };
}

/** Feed frames in order, collecting everything each one produced. */
function run(events: ChainJobEvent[]): {
  progress: ChainJobProgressFrame[];
  last: ChainJobFrameResult;
} {
  let live = emptyChainJobLive();
  const progress: ChainJobProgressFrame[] = [];
  let last!: ChainJobFrameResult;
  for (const event of events) {
    last = reduceChainJobFrame(live, event);
    live = last.live;
    progress.push(...last.progress);
  }
  return { progress, last };
}

describe("reduceChainJobFrame", () => {
  it("opens with chain_start built from the snapshot, not a synthesized frame", () => {
    const { progress } = run([{ type: "snapshot", job: detail() }]);
    expect(progress).toEqual([
      { type: "chain_start", stage_count: 2, estimated_total_frames: 194 },
    ]);
  });

  it("estimates zero frames when the job declares no script", () => {
    const { progress } = run([
      { type: "snapshot", job: detail({ script: null }) },
    ]);
    expect(progress[0]).toMatchObject({ estimated_total_frames: 0 });
  });

  it("passes stage and denoise frames through unchanged", () => {
    const { progress } = run([
      { type: "snapshot", job: detail() },
      { type: "stage_start", stage_idx: 0 },
      { type: "denoise_step", stage_idx: 0, step: 3, total: 8 },
    ]);
    expect(progress.slice(1)).toEqual([
      { type: "stage_start", stage_idx: 0 },
      { type: "denoise_step", stage_idx: 0, step: 3, total: 8 },
    ]);
  });

  /**
   * The durable `stage_done` carries no frame count, so it is read back off
   * the snapshot rather than invented — a caller that guessed would report a
   * clip length the machine never emitted.
   */
  it("reads frames_emitted for stage_done off the maintained snapshot", () => {
    const withFrames = detail();
    withFrames.stages[0]!.frames_emitted = 97;
    const { progress } = run([
      { type: "snapshot", job: withFrames },
      { type: "stage_done", stage_idx: 0, has_media: true },
    ]);
    expect(progress.at(-1)).toEqual({
      type: "stage_done",
      stage_idx: 0,
      frames_emitted: 97,
    });
  });

  it("reports zero rather than guessing when the snapshot has no count", () => {
    const { progress } = run([
      { type: "snapshot", job: detail() },
      { type: "stage_done", stage_idx: 1 },
    ]);
    expect(progress.at(-1)).toMatchObject({ frames_emitted: 0 });
  });

  it("maps finalizing to stitching, falling back to the declared frames", () => {
    const explicit = run([
      { type: "snapshot", job: detail() },
      { type: "finalizing", total_frames: 190 },
    ]);
    expect(explicit.progress.at(-1)).toEqual({
      type: "stitching",
      total_frames: 190,
    });
    const implied = run([
      { type: "snapshot", job: detail() },
      { type: "finalizing" },
    ]);
    expect(implied.progress.at(-1)).toEqual({
      type: "stitching",
      total_frames: 194,
    });
  });

  /** Completion is a saved FILENAME; the caller fetches media from the gallery. */
  it("surfaces the finalized filename and never a progress frame", () => {
    const { last, progress } = run([
      { type: "snapshot", job: detail() },
      {
        type: "finalized",
        output: "final/output-2.mp4",
        gallery_filename: "stitched.mp4",
        take: 2,
      },
    ]);
    expect(last.finalized).toEqual({ output: "stitched.mp4", take: 2 });
    expect(progress).toHaveLength(1);
  });

  it("settles only on a terminal state, and carries the machine's reason", () => {
    for (const [state, error] of [
      ["completed", null],
      ["failed", "host ran out of memory"],
      ["cancelled", null],
    ] as const) {
      const { last } = run([
        { type: "snapshot", job: detail() },
        { type: "state_changed", state, error },
      ]);
      expect(last.terminal).toEqual({ state, error });
    }
    for (const state of ["queued", "running", "interrupted"] as const) {
      const { last } = run([
        { type: "snapshot", job: detail() },
        { type: "state_changed", state },
      ]);
      expect(last.terminal).toBeNull();
    }
  });

  it("settles a job that was already terminal when the stream attached", () => {
    const { last } = run([
      { type: "snapshot", job: detail({ state: "failed", error: "gone" }) },
    ]);
    expect(last.terminal).toEqual({ state: "failed", error: "gone" });
  });

  it("produces nothing for a frame with no surface meaning", () => {
    const { last } = run([
      { type: "snapshot", job: detail() },
      { type: "yielded" },
    ]);
    expect(last.progress).toEqual([]);
    expect(last.finalized).toBeNull();
    expect(last.terminal).toBeNull();
  });

  it("keeps the live snapshot as the one authority the caller reads", () => {
    const { last } = run([
      { type: "snapshot", job: detail() },
      { type: "stage_start", stage_idx: 1 },
    ]);
    expect(last.live.activeStage).toBe(1);
    expect(estimatedChainFrames(last.live)).toBe(194);
  });
});
