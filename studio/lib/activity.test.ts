import { describe, expect, it } from "vitest";
import {
  activityAnnouncement,
  activityCountLabel,
  activityDigestLabel,
  mergeActivity,
  partitionActivity,
  sequenceActionLabel,
  sequenceActions,
  sequenceToVM,
  queueStatusLabel,
  withLiveQueueStatus,
  MAX_ATTENTION_ROWS,
  SETTLED_VISIBLE_MS,
  type ActivityJobVM,
  type PrintActivityVM,
} from "./activity";
import {
  compareNewestQueueEntry,
  compareNewestSubmitted,
} from "./activityOrder";
import { buildQueueStatusIndex } from "./queuePosition";
import type { ChainJobSummary } from "./api/chainTypes";

function print(
  extra: Partial<ActivityJobVM & { kind: "print" }> = {},
): ActivityJobVM {
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

describe("compareNewestSubmitted", () => {
  it("sorts newest-first and preserves sending order for equal timestamps", () => {
    const rows = [
      { id: "equal-first", createdAtMs: 20 },
      { id: "oldest", createdAtMs: 10 },
      { id: "equal-second", createdAtMs: 20 },
      { id: "newest", createdAtMs: 30 },
    ];
    expect(rows.sort(compareNewestSubmitted).map(({ id }) => id)).toEqual([
      "newest",
      "equal-first",
      "equal-second",
      "oldest",
    ]);
  });

  it("sorts queue entries newest-first without using scheduler position", () => {
    const entries = [
      { id: "sent-first", started_at_unix_ms: 20, position: 9 },
      { id: "oldest", started_at_unix_ms: 10, position: 0 },
      { id: "sent-second", started_at_unix_ms: 20, position: 1 },
      { id: "newest", started_at_unix_ms: 30, position: 7 },
    ];
    expect(entries.sort(compareNewestQueueEntry).map(({ id }) => id)).toEqual([
      "newest",
      "sent-first",
      "sent-second",
      "oldest",
    ]);
  });
});

describe("sequenceActions", () => {
  it("gives running jobs cancel + watch, settled jobs edit paths", () => {
    expect(sequenceActions("queued")).toEqual(["watch", "cancel"]);
    expect(sequenceActions("running")).toEqual(["watch", "cancel"]);
    expect(sequenceActions("completed")).toEqual(["watch", "edit", "delete"]);
    // Resumability is a server feature the strip must surface.
    expect(sequenceActions("interrupted")).toEqual([
      "resume",
      "edit",
      "delete",
    ]);
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

  it("keeps a claimed parent queued until the server reports a leased stage", () => {
    const waiting = sequenceToVM(
      summary({ state: "running", execution_phase: "queued" }),
      { hostId: "plato", hostLabel: "plato" },
    );
    const leased = sequenceToVM(
      summary({ state: "running", execution_phase: "running" }),
      { hostId: "plato", hostLabel: "plato" },
    );
    if (waiting.kind !== "sequence" || leased.kind !== "sequence") return;
    expect(waiting.phase).toBe("queued");
    expect(leased.phase).toBe("running");
    expect(mergeActivity([], [waiting, leased]).map((vm) => vm.key)).toEqual([
      waiting.key,
      leased.key,
    ]);
  });
});

describe("mergeActivity", () => {
  it("keeps active work newest-first regardless of phase, then settled by recency", () => {
    const settledPrint = print({
      key: "print:1",
      phase: "done",
      createdAtMs: 400,
    });
    const runningPrint = print({
      key: "print:2",
      phase: "running",
      createdAtMs: 100,
    });
    const queuedSeq = sequenceToVM(summary({ created_at_unix_ms: 300 }), {
      hostId: "plato",
      hostLabel: "plato",
    });
    const settledSeq = sequenceToVM(
      summary({ id: "c2", state: "completed", created_at_unix_ms: 500 }),
      { hostId: "plato", hostLabel: "plato" },
    );

    const merged = mergeActivity(
      [settledPrint, runningPrint],
      [queuedSeq, settledSeq],
    );
    expect(merged.map((vm) => vm.key)).toEqual([
      "seq:plato:c1", // newer queued work stays above older running work
      "print:2",
      "seq:plato:c2", // then settled, newest first
      "print:1",
    ]);
  });

  // Regression: printVMs used to carry `createdAtMs = job.clientId` (a 1,2,3…
  // counter). Harmless while every print VM was filtered back out after the
  // merge, fatal the moment a failed print keeps a row — a counter always
  // loses to a ~1.7e12 epoch stamp and sorts to the bottom forever.
  it("sorts prints against sequences on a real wall clock", () => {
    const freshPrint = print({
      key: "print:new",
      phase: "failed",
      createdAtMs: 1_700_000_100_000,
    });
    const olderSeq = sequenceToVM(
      summary({
        id: "old",
        state: "failed",
        created_at_unix_ms: 1_700_000_000_000,
      }),
      { hostId: "plato", hostLabel: "plato" },
    );
    expect(mergeActivity([freshPrint], [olderSeq]).map((vm) => vm.key)).toEqual(
      ["print:new", "seq:plato:old"],
    );
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
    const rows = [
      seq({ id: "q", state: "queued" }),
      seq({ id: "r", state: "running" }),
    ];
    const part = partitionActivity(rows, { nowMs: NOW });
    expect(part.active.map((vm) => vm.key)).toEqual([
      "seq:plato:q",
      "seq:plato:r",
    ]);
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
      [
        seq({
          id: "f",
          state: "failed",
          updated_at_unix_ms: NOW - SETTLED_VISIBLE_MS - 1,
        }),
      ],
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
        print({
          key: "print:9",
          phase: "failed",
          createdAtMs: NOW,
          settledAtMs: NOW - 5_000,
        }),
        seq({ id: "f", state: "failed", updated_at_unix_ms: NOW - 1_000 }),
      ],
      { nowMs: NOW },
    );
    expect(part.attention.map((vm) => vm.key)).toEqual([
      "seq:plato:f",
      "print:9",
    ]);
  });

  it("keeps the newest rows at the cap and reports the overflow", () => {
    const rows = [
      seq({ id: "a", state: "failed", updated_at_unix_ms: NOW - 3_000 }),
      seq({ id: "b", state: "failed", updated_at_unix_ms: NOW - 2_000 }),
      seq({ id: "c", state: "failed", updated_at_unix_ms: NOW - 1_000 }),
    ];
    const part = partitionActivity(rows, {
      nowMs: NOW,
      maxAttentionRows: MAX_ATTENTION_ROWS,
    });
    expect(part.attention.map((vm) => vm.key)).toEqual([
      "seq:plato:c",
      "seq:plato:b",
    ]);
    expect(part.hiddenAttention).toBe(1);
    // Overflow is a failure count, not a settled-sequence count.
    expect(part.settledSequences).toBe(0);
  });

  it("drops dismissed rows without moving them into the settled count", () => {
    const rows = [
      seq({ id: "f", state: "failed", updated_at_unix_ms: NOW - 1_000 }),
    ];
    const kept = partitionActivity(rows, { nowMs: NOW });
    const dropped = partitionActivity(rows, {
      nowMs: NOW,
      dismissed: ["seq:plato:f"],
    });
    expect(kept.attention).toHaveLength(1);
    expect(dropped.attention).toEqual([]);
    expect(dropped.hiddenAttention).toBe(0);
    expect(dropped.settledSequences).toBe(kept.settledSequences);
  });
});

describe("activityDigestLabel", () => {
  it("stays silent when the strip is showing everything", () => {
    expect(
      activityDigestLabel({ settledSequences: 0, hiddenAttention: 0 }),
    ).toBeNull();
  });

  it("names settled sequences, hidden failures, or both", () => {
    expect(
      activityDigestLabel({ settledSequences: 4, hiddenAttention: 0 }),
    ).toBe("4 settled sequences");
    expect(
      activityDigestLabel({ settledSequences: 1, hiddenAttention: 0 }),
    ).toBe("1 settled sequence");
    expect(
      activityDigestLabel({ settledSequences: 4, hiddenAttention: 1 }),
    ).toBe("1 failed · 4 settled sequences");
    expect(
      activityDigestLabel({ settledSequences: 0, hiddenAttention: 3 }),
    ).toBe("3 failed");
  });
});

describe("withLiveQueueStatus", () => {
  const index = buildQueueStatusIndex([
    {
      hostId: "local",
      entries: [
        {
          id: "srv-run",
          model: "m",
          state: "running",
          started_at_unix_ms: 1,
          position: 0,
        },
        {
          id: "srv-2",
          model: "m",
          state: "queued",
          started_at_unix_ms: 2,
          position: 2,
        },
        {
          id: "srv-3",
          model: "m",
          state: "queued",
          started_at_unix_ms: 3,
          position: 3,
        },
      ],
      plan: {
        plan_version: 1,
        state_version: 1,
        optimizer_state: "settled",
        dirty_since_unix_ms: null,
        next_replan_at_unix_ms: null,
        work_items: [
          {
            work_id: "w3",
            parent_id: "srv-3",
            work_kind: "generation",
            priority_class: "normal",
            queue_rank: 3,
            bypass_count: 0,
            estimate_confidence: "low",
            blocked_reason: "insufficient_host_ram",
          },
        ],
      },
    },
  ]);

  it("maps a queued job to its live position", () => {
    const vm = withLiveQueueStatus(print() as PrintActivityVM, index, "srv-2");
    expect(vm.queuePosition).toBe(2);
    expect(queueStatusLabel(vm)).toBe("#2 in line");
  });

  it("leaves the position absent when the host never listed the job", () => {
    const vm = withLiveQueueStatus(
      print() as PrintActivityVM,
      index,
      "srv-missing",
    );
    expect(vm.queuePosition).toBeUndefined();
    // No evidence still says the one true thing: it is queued.
    expect(queueStatusLabel(vm)).toBe("Queued");
    const unread = withLiveQueueStatus(
      print() as PrintActivityVM,
      null,
      "srv-2",
    );
    expect(unread.queuePosition).toBeUndefined();
    const unsubmitted = withLiveQueueStatus(
      print() as PrintActivityVM,
      index,
      "",
    );
    expect(unsubmitted.queuePosition).toBeUndefined();
  });

  it("never carries a position on running or settled rows", () => {
    for (const phase of ["running", "done", "failed", "cancelled"] as const) {
      const vm = withLiveQueueStatus(
        print({
          phase,
          settledAtMs: phase === "running" ? null : 10,
        }) as PrintActivityVM,
        index,
        "srv-2",
      );
      expect(vm.queuePosition).toBeUndefined();
      expect(queueStatusLabel(vm)).toBeNull();
    }
  });

  it("says why a parked job is waiting instead of counting its place", () => {
    const vm = withLiveQueueStatus(print() as PrintActivityVM, index, "srv-3");
    expect(vm.blockedReason).toBe("insufficient_host_ram");
    expect(queueStatusLabel(vm)).toBe("Waiting for memory");
  });

  it("names the job at the head of the queue", () => {
    const vm = withLiveQueueStatus(
      print() as PrintActivityVM,
      index,
      "srv-run",
    );
    expect(vm.queuePosition).toBe(0);
    expect(queueStatusLabel(vm)).toBe("Next up");
  });

  it("says nothing for a sequence row", () => {
    expect(queueStatusLabel(print({ kind: "sequence" } as never))).toBeNull();
  });
});

describe("activity counts", () => {
  it("never calls queued work active", () => {
    // The iPhone screenshot: one z-image job running, four behind it.
    expect(activityCountLabel({ running: 1, waiting: 4 })).toBe(
      "1 active · 4 queued",
    );
    expect(activityAnnouncement({ running: 1, waiting: 4 })).toBe(
      "1 active generation, 4 queued.",
    );
  });

  it("drops the half that is zero", () => {
    expect(activityCountLabel({ running: 2, waiting: 0 })).toBe("2 active");
    expect(activityCountLabel({ running: 0, waiting: 3 })).toBe("3 queued");
    expect(activityAnnouncement({ running: 1, waiting: 0 })).toBe(
      "1 active generation.",
    );
    expect(activityAnnouncement({ running: 0, waiting: 1 })).toBe(
      "1 queued generation.",
    );
    expect(activityAnnouncement({ running: 0, waiting: 3 })).toBe(
      "3 queued generations.",
    );
  });

  it("stays honest when only settled rows are left on screen", () => {
    expect(activityCountLabel({ running: 0, waiting: 0 })).toBe("0 active");
    expect(activityAnnouncement({ running: 0, waiting: 0 })).toBe(
      "No active generations.",
    );
  });
});
