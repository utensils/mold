import { describe, expect, it } from "vitest";
import { applyChainJobEvent, emptyChainJobLive } from "./chainJobs";
import type { ChainJobDetail, ChainJobEvent } from "../lib/api/types";

function detail(): ChainJobDetail {
  return {
    id: "job-1",
    state: "running",
    model: "ltx-2-19b-distilled:fp8",
    stage_count: 2,
    current_stage: 0,
    created_at_unix_ms: 1,
    updated_at_unix_ms: 2,
    error: null,
    ephemeral: false,
    stages: [
      { idx: 0, state: "pending", seed: "42", has_preview: false },
      { idx: 1, state: "pending", seed: "43", has_preview: false },
    ],
    finalizes: [],
    retakes: [],
    script: { schema: "mold.chain.v1", chain: {} as never, stage: [] },
  };
}

function reduce(events: ChainJobEvent[]) {
  return events.reduce(applyChainJobEvent, emptyChainJobLive());
}

describe("applyChainJobEvent", () => {
  it("seeds the live view from the snapshot frame", () => {
    const s = reduce([{ type: "snapshot", job: detail() }]);
    expect(s.detail?.id).toBe("job-1");
    expect(s.activeStage).toBeNull();
    expect(s.progress).toEqual({});
  });

  it("marks a stage running on stage_start and tracks the active stage", () => {
    const s = reduce([
      { type: "snapshot", job: detail() },
      { type: "stage_start", stage_idx: 0 },
    ]);
    expect(s.activeStage).toBe(0);
    expect(s.detail?.stages[0]!.state).toBe("running");
  });

  it("records per-stage denoise progress", () => {
    const s = reduce([
      { type: "snapshot", job: detail() },
      { type: "stage_start", stage_idx: 0 },
      { type: "denoise_step", stage_idx: 0, step: 4, total: 8 },
    ]);
    expect(s.progress[0]).toEqual({ step: 4, total: 8 });
  });

  it("completes a stage and clears the active stage on stage_done", () => {
    const s = reduce([
      { type: "snapshot", job: detail() },
      { type: "stage_start", stage_idx: 0 },
      { type: "denoise_step", stage_idx: 0, step: 8, total: 8 },
      { type: "stage_done", stage_idx: 0, frames_emitted: 97, has_preview: true },
    ]);
    expect(s.detail?.stages[0]!.state).toBe("completed");
    expect(s.detail?.stages[0]!.has_preview).toBe(true);
    expect(s.activeStage).toBeNull();
  });

  it("propagates the terminal state on state_changed", () => {
    const s = reduce([
      { type: "snapshot", job: detail() },
      { type: "state_changed", state: "completed", error: null },
    ]);
    expect(s.detail?.state).toBe("completed");
  });

  it("snapshot resumes an in-flight job with the running stage active", () => {
    const d = detail();
    d.stages[0]!.state = "completed";
    d.stages[1]!.state = "running";
    const s = reduce([{ type: "snapshot", job: d }]);
    expect(s.activeStage).toBe(1);
  });
});
