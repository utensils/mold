import { describe, expect, it } from "vitest";
import {
  activityDigestLabel,
  mergeActivity,
  partitionActivity,
  sequenceActionLabel,
  sequenceActions,
  sequenceToVM,
  MAX_ATTENTION_ROWS,
  SETTLED_VISIBLE_MS,
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
    settledAtMs: null,
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

  // Regression: printVMs used to carry `createdAtMs = job.clientId` (a 1,2,3…
  // counter). Harmless while every print VM was filtered back out after the
  // merge, fatal the moment a failed print keeps a row — a counter always
  // loses to a ~1.7e12 epoch stamp and sorts to the bottom forever.
  it("sorts prints against sequences on a real wall clock", () => {
    const freshPrint = print({ key: "print:new", phase: "failed", createdAtMs: 1_700_000_100_000 });
    const olderSeq = sequenceToVM(
      summary({ id: "old", state: "failed", created_at_unix_ms: 1_700_000_000_000 }),
      { hostId: "plato", hostLabel: "plato" },
    );
    expect(mergeActivity([freshPrint], [olderSeq]).map((vm) => vm.key)).toEqual([
      "print:new",
      "seq:plato:old",
    ]);
  });
});

describe("sequenceToVM settle stamps", () => {
  it("stamps settledAtMs from updated_at_unix_ms only once a job settles", () => {
    const forState = (state: ChainJobSummary["state"]) =>
      sequenceToVM(summary({ state, updated_at_unix_ms: 4242 }), {
        hostId: "plato",
        hostLabel: "plato",
      }).settledAtMs;
    expect(forState("queued")).toBeNull();
    expect(forState("running")).toBeNull();
    expect(forState("completed")).toBe(4242);
    expect(forState("failed")).toBe(4242);
    expect(forState("cancelled")).toBe(4242);
    expect(forState("interrupted")).toBe(4242);
  });
});

describe("sequenceActionLabel", () => {
  it("watches present-tense work and opens settled work", () => {
    expect(sequenceActionLabel("watch", "running")).toBe("Watch");
    expect(sequenceActionLabel("watch", "queued")).toBe("Watch");
    expect(sequenceActionLabel("watch", "completed")).toBe("Open");
    expect(sequenceActionLabel("watch", "failed")).toBe("Open");
    expect(sequenceActionLabel("resume", "failed")).toBe("Resume");
  });
});

// "Activity is present tense": the strip renders in-flight work plus a capped,
// expiring set of settled-but-wrong rows. Everything else becomes a count.
describe("partitionActivity", () => {
  const NOW = 1_700_000_000_000;
  const seq = (extra: Partial<ChainJobSummary> = {}, hostId = "plato") =>
    sequenceToVM(summary(extra), { hostId, hostLabel: hostId });

  it("puts queued and running rows in active and never in attention", () => {
    const rows = [seq({ id: "q", state: "queued" }), seq({ id: "r", state: "running" })];
    const part = partitionActivity(rows, { nowMs: NOW });
    expect(part.active.map((vm) => vm.key)).toEqual(["seq:plato:q", "seq:plato:r"]);
    expect(part.attention).toEqual([]);
    expect(part.settledSequences).toBe(0);
  });

  it("counts a completed sequence once and renders no row for it", () => {
    const part = partitionActivity(
      [seq({ id: "done", state: "completed", updated_at_unix_ms: NOW - 1000 })],
      { nowMs: NOW },
    );
    expect(part.active).toEqual([]);
    expect(part.attention).toEqual([]);
    expect(part.settledSequences).toBe(1);
  });

  it("never gives a cancelled sequence an attention row", () => {
    const part = partitionActivity(
      [seq({ id: "x", state: "cancelled", updated_at_unix_ms: NOW - 1000 })],
      { nowMs: NOW },
    );
    expect(part.attention).toEqual([]);
    expect(part.settledSequences).toBe(1);
  });

  it("keeps a fresh failure and ages out a stale one into the digest", () => {
    const fresh = partitionActivity(
      [seq({ id: "f", state: "failed", updated_at_unix_ms: NOW - 60_000 })],
      { nowMs: NOW },
    );
    expect(fresh.attention.map((vm) => vm.key)).toEqual(["seq:plato:f"]);
    expect(fresh.settledSequences).toBe(0);

    const stale = partitionActivity(
      [seq({ id: "f", state: "failed", updated_at_unix_ms: NOW - SETTLED_VISIBLE_MS - 1 })],
      { nowMs: NOW },
    );
    expect(stale.attention).toEqual([]);
    expect(stale.settledSequences).toBe(1);
  });

  it("treats an interrupted sequence as attention (it is resumable)", () => {
    const part = partitionActivity(
      [seq({ id: "i", state: "interrupted", updated_at_unix_ms: NOW - 1000 })],
      { nowMs: NOW },
    );
    expect(part.attention.map((vm) => vm.key)).toEqual(["seq:plato:i"]);
  });

  it("gives a failed print its own attention row, keyed distinctly", () => {
    const part = partitionActivity(
      [
        print({ key: "print:9", phase: "failed", createdAtMs: NOW, settledAtMs: NOW - 5_000 }),
        seq({ id: "f", state: "failed", updated_at_unix_ms: NOW - 1_000 }),
      ],
      { nowMs: NOW },
    );
    expect(part.attention.map((vm) => vm.key)).toEqual(["seq:plato:f", "print:9"]);
  });

  it("keeps the newest rows at the cap and reports the overflow", () => {
    const rows = [
      seq({ id: "a", state: "failed", updated_at_unix_ms: NOW - 3_000 }),
      seq({ id: "b", state: "failed", updated_at_unix_ms: NOW - 2_000 }),
      seq({ id: "c", state: "failed", updated_at_unix_ms: NOW - 1_000 }),
    ];
    const part = partitionActivity(rows, { nowMs: NOW, maxAttentionRows: MAX_ATTENTION_ROWS });
    expect(part.attention.map((vm) => vm.key)).toEqual(["seq:plato:c", "seq:plato:b"]);
    expect(part.hiddenAttention).toBe(1);
    // Overflow is a failure count, not a settled-sequence count.
    expect(part.settledSequences).toBe(0);
  });

  it("drops dismissed rows without moving them into the settled count", () => {
    const rows = [seq({ id: "f", state: "failed", updated_at_unix_ms: NOW - 1_000 })];
    const kept = partitionActivity(rows, { nowMs: NOW });
    const dropped = partitionActivity(rows, { nowMs: NOW, dismissed: ["seq:plato:f"] });
    expect(kept.attention).toHaveLength(1);
    expect(dropped.attention).toEqual([]);
    expect(dropped.hiddenAttention).toBe(0);
    expect(dropped.settledSequences).toBe(kept.settledSequences);
  });
});

describe("activityDigestLabel", () => {
  it("stays silent when the strip is showing everything", () => {
    expect(activityDigestLabel({ settledSequences: 0, hiddenAttention: 0 })).toBeNull();
  });

  it("names settled sequences, hidden failures, or both", () => {
    expect(activityDigestLabel({ settledSequences: 4, hiddenAttention: 0 })).toBe(
      "4 settled sequences",
    );
    expect(activityDigestLabel({ settledSequences: 1, hiddenAttention: 0 })).toBe(
      "1 settled sequence",
    );
    expect(activityDigestLabel({ settledSequences: 4, hiddenAttention: 1 })).toBe(
      "1 failed · 4 settled sequences",
    );
    expect(activityDigestLabel({ settledSequences: 0, hiddenAttention: 3 })).toBe("3 failed");
  });
});
