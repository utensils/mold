import { describe, expect, it } from "vitest";
import {
  mergeActivity,
  sequenceActions,
  sequenceToVM,
  type ActivityJobVM,
} from "./activity";
import type { ChainJobSummary } from "./api/chainTypes";

function print(extra: Partial<ActivityJobVM & { kind: "print" }> = {}): ActivityJobVM {
  return {
    kind: "print",
    key: "print:1",
    hostId: "local",
    hostLabel: "This device",
    model: "flux-dev:q8",
    prompt: "a cat",
    phase: "queued",
    progress: null,
    chain: null,
    actions: ["cancel"],
    createdAtMs: 100,
    ...extra,
  } as ActivityJobVM;
}

function summary(extra: Partial<ChainJobSummary> = {}): ChainJobSummary {
  return {
    id: "c1",
    state: "queued",
    model: "ltx-2-19b-distilled:fp8",
    stage_count: 2,
    current_stage: 0,
    created_at_unix_ms: 50,
    updated_at_unix_ms: 60,
    error: null,
    ...extra,
  };
}

describe("sequenceActions", () => {
  it("gives running jobs cancel + watch, settled jobs edit paths", () => {
    expect(sequenceActions("queued")).toEqual(["watch", "cancel"]);
    expect(sequenceActions("running")).toEqual(["watch", "cancel"]);
    expect(sequenceActions("completed")).toEqual(["watch", "edit", "delete"]);
    // Resumability is a server feature the strip must surface.
    expect(sequenceActions("interrupted")).toEqual(["resume", "edit", "delete"]);
    expect(sequenceActions("failed")).toEqual(["resume", "edit", "delete"]);
    expect(sequenceActions("cancelled")).toEqual(["resume", "edit", "delete"]);
  });
});

describe("sequenceToVM", () => {
  it("builds a sequence row with friendly error text", () => {
    const vm = sequenceToVM(
      summary({
        state: "failed",
        error: 'DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")',
      }),
      { hostId: "plato", hostLabel: "plato" },
    );
    expect(vm.kind).toBe("sequence");
    if (vm.kind !== "sequence") return;
    expect(vm.key).toBe("seq:plato:c1");
    expect(vm.stageCount).toBe(2);
    expect(vm.error).toContain("GPU memory");
    expect(vm.actions).toEqual(["resume", "edit", "delete"]);
  });

  it("carries live denoise progress for the watched job", () => {
    const vm = sequenceToVM(
      summary({ state: "running", current_stage: 1 }),
      { hostId: "plato", hostLabel: "plato" },
      { step: 4, total: 8 },
    );
    if (vm.kind !== "sequence") return;
    expect(vm.progress).toEqual({ step: 4, total: 8 });
  });
});

describe("mergeActivity", () => {
  it("orders active work first, then by recency", () => {
    const settledPrint = print({ key: "print:1", phase: "done", createdAtMs: 400 });
    const runningPrint = print({ key: "print:2", phase: "running", createdAtMs: 100 });
    const queuedSeq = sequenceToVM(summary({ created_at_unix_ms: 300 }), {
      hostId: "plato",
      hostLabel: "plato",
    });
    const settledSeq = sequenceToVM(
      summary({ id: "c2", state: "completed", created_at_unix_ms: 500 }),
      { hostId: "plato", hostLabel: "plato" },
    );

    const merged = mergeActivity([settledPrint, runningPrint], [queuedSeq, settledSeq]);
    expect(merged.map((vm) => vm.key)).toEqual([
      "print:2", // running first
      "seq:plato:c1", // queued second
      "seq:plato:c2", // then settled, newest first
      "print:1",
    ]);
  });
});
